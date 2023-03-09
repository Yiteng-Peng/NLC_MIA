from coverage.Coverage import *
import coverage.tool as tool


class KMNC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.name = 'KMNC'

        self.k = hyper
        self.range_dict = {}

        coverage_multisec_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            coverage_multisec_dict[layer_name] = torch.zeros((num_neuron, self.k + 1))\
                                                      .type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000,
                                           torch.ones(num_neuron).to(self.device) * -10000]

        self.coverage_dict = {'multisec': coverage_multisec_dict}
        self.current = 0

    def build(self, data_loader):
        print('Building range...')
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.set_range(data)

    def set_range(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        multisec_cove_dict = {}
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            multisec_index = (u_bound > l_bound) & (layer_output >= l_bound) & (layer_output <= u_bound)
            multisec_covered = torch.zeros(num_neuron, self.k + 1).type(torch.BoolTensor).to(self.device)
            div_index = u_bound > l_bound
            div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
            multisec_output = torch.ceil((layer_output - l_bound) / div * self.k).type(torch.LongTensor).to(
                self.device) * multisec_index
            # (1, k), index 0 indicates out-of-range output

            index = tuple([torch.LongTensor(list(range(num_neuron))), multisec_output])
            multisec_covered[index] = True
            multisec_cove_dict[layer_name] = multisec_covered | self.cove_dict['multisec'][layer_name]

        return {
            'multisec': multisec_cove_dict
        }

    def coverage(self, cove_dict):
        multisec_cove_dict = cove_dict['multisec']
        (multisec_cove, multisec_total) = (0, 0)
        for layer_name in multisec_cove_dict.keys():
            multisec_covered = multisec_cove_dict[layer_name]
            num_neuron = multisec_covered.size(0)
            multisec_cove += torch.sum(multisec_covered[:, 1:])
            multisec_total += (num_neuron * self.k)
        multisec_rate = multisec_cove / multisec_total
        return multisec_rate.item()
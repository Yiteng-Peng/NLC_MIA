from coverage.Coverage import *
import coverage.tool as tool
from coverage.KMNC import KMNC

class SNAC(KMNC):
    def init_variable(self, hyper=None):
        assert hyper is None
        self.name = 'SNAC'
        self.range_dict = {}
        coverage_upper_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            coverage_upper_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000, torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'upper': coverage_upper_dict
        }
        self.current = 0

    def calculate(self, data):
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            upper_covered = (layer_output > u_bound).sum(0) > 0
            upper_cove_dict[layer_name] = upper_covered | self.coverage_dict['upper'][layer_name]

        return {
            'upper': upper_cove_dict
        }

    def coverage(self, cove_dict):
        upper_cove_dict = cove_dict['upper']
        (upper_cove, upper_total) = (0, 0)
        for layer_name in upper_cove_dict.keys():
            upper_covered = upper_cove_dict[layer_name]
            upper_cove += upper_covered.sum()
            upper_total += len(upper_covered)
        upper_rate = upper_cove / upper_total
        return upper_rate.item()

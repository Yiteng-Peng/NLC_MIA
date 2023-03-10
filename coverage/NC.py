from coverage.Coverage import *
import coverage.tool as tool


class NC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.name = "NC"

        self.threshold = hyper

        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)

        self.current = 0

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            scaled_output = tool.scale(layer_output)
            mask_index = scaled_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()
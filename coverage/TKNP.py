from coverage.Coverage import *
import coverage.tool as tool


class TKNP(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.k = hyper
        self.layer_pattern = {}
        self.network_pattern = set()
        self.current = 0
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.layer_pattern[layer_name] = set()
        self.coverage_dict = {
            'layer_pattern': self.layer_pattern,
            'network_pattern': self.network_pattern
        }

    def calculate(self, data):
        layer_pat = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        topk_idx_list = []
        for (layer_name, layer_output) in layer_output_dict.items():
            num_neuron = layer_output.size(1)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=True)
            # idx: (batch_size, k)
            pat = set([str(s) for s in list(idx[:, ])])
            topk_idx_list.append(idx)
            layer_pat[layer_name] = set.union(pat, self.layer_pattern[layer_name])
        network_topk_idx = torch.cat(topk_idx_list, 1)
        network_pat = set([str(s) for s in list(network_topk_idx[:, ])])
        network_pat = set.union(network_pat, self.network_pattern)
        return {
            'layer_pattern': layer_pat,
            'network_pattern': network_pat
        }

    def coverage(self, cove_dict, mode='network'):
        assert mode in ['network', 'layer']
        if mode == 'network':
            return len(cove_dict['network_pattern'])
        cnt = 0
        if mode == 'layer':
            for layer_name in cove_dict['layer_pattern'].keys():
                cnt += len(cove_dict['layer_pattern'][layer_name])
        return cnt

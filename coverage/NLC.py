from coverage.Coverage import *
import coverage.tool as tool


class NLC(Coverage):
    def init_variable(self, hyper=None):
        assert hyper is None, 'NLC has no hyper-parameter'
        self.estimator_dict = {}
        self.current = 1
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = tool.Estimator(feature_num=layer_size[0])

    def calculate(self, data):
        stat_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            info_dict = self.estimator_dict[layer_name].calculate(layer_output.to(self.device))
            stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])
        return stat_dict

    def update(self, stat_dict, gain=None):
        if gain is None:
            for i, layer_name in enumerate(stat_dict.keys()):
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current = self.coverage(self.estimator_dict)
        else:
            (delta, layer_to_update) = gain
            for layer_name in layer_to_update:
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current += delta

    def coverage(self, stat_dict):
        val = 0
        for i, layer_name in enumerate(stat_dict.keys()):
            # Ave = stat_dict[layer_name].Ave
            CoVariance = stat_dict[layer_name].CoVariance
            # Amount = stat_dict[layer_name].Amount
            val += self.norm(CoVariance)
        return val

    def gain(self, stat_new):
        total = 0
        layer_to_update = []
        for i, layer_name in enumerate(stat_new.keys()):
            (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
            value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance)
            if value > 0:
                layer_to_update.append(layer_name)
                total += value
        if total > 0:
            return (total, layer_to_update)
        else:
            return None

    def norm(self, vec, mode='L1', reduction='mean'):
        m = vec.size(0)
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = vec.abs().sum()
        elif mode == 'L2':
            total = vec.pow(2).sum().sqrt()
        if reduction == 'mean':
            return total / m
        elif reduction == 'sum':
            return total

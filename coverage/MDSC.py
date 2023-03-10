from coverage.Coverage import *
import coverage.tool as tool


class MDSC(SurpriseCoverage):
    def get_name(self):
        return 'MDSC'

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)
        self.estimator = tool.Estimator(feature_num=feature_num, num_class=self.num_class)

    def build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        # print('SA_batch: ', SA_batch.size())
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        stat_dict = self.estimator.calculate(SA_batch, label_batch)
        self.estimator.update(stat_dict)

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        mu = self.estimator.Ave[label_batch]
        covar = self.estimator.CoVariance[label_batch]
        covar_inv = torch.linalg.inv(covar)
        mdsa = (
            torch.bmm(torch.bmm((SA_batch - mu).unsqueeze(1), covar_inv),
            (SA_batch - mu).unsqueeze(2))
        ).sqrt()
        # [bs, 1, n] x [bs, n, n] x [bs, n, 1]
        # [bs, 1]
        mdsa = mdsa.view(batch_size, -1)
        mdsa = mdsa[~torch.any(mdsa.isnan(), dim=1)]
        mdsa = mdsa[~torch.any(mdsa.isinf(), dim=1)]
        mdsa = mdsa.view(-1)
        if len(mdsa) > 0:
            mdsa_list = (mdsa / self.threshold).cpu().numpy().tolist()
            mdsa_list = [int(_mdsa) for _mdsa in mdsa_list]
            cove_set = set(mdsa_list)
            cove_set = self.coverage_set.union(cove_set)
        return cove_set

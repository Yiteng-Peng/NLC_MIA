from coverage.Coverage import *
import coverage.tool as tool


class LSC(SurpriseCoverage):
    def get_name(self):
        return 'LSC'

    def set_kde(self):
        for k in self.SA_cache.keys():
            if self.num_class <= 1:
                self.kde_cache[k] = gaussian_kde(self.SA_cache[k].T)
            else:
                self.kde_cache[k] = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.SA_cache[k])
            # The original LSC uses the `gaussian_kde` function, however, we note that this function
            # frequently crashes due to numerical issues, especially for large `num_class`.

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy() # [batch_size, num_neuron]
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]
            # if (np.isnan(SA).any()) or (not np.isinf(SA).any()):
            #     continue
            if self.num_class <= 1:
                lsa = np.asscalar(-self.kde_cache[int(label.cpu())].logpdf(np.expand_dims(SA, 1)))
            else:
                lsa = np.asscalar(-self.kde_cache[int(label.cpu())].score_samples(np.expand_dims(SA, 0)))
            if (not np.isnan(lsa)) and (not np.isinf(lsa)):
                cove_set.add(int(lsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        return cove_set

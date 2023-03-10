from coverage.Coverage import *
import coverage.tool as tool


class DSC(SurpriseCoverage):
    def get_name(self):
        return 'DSC'

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy()  # [batch_size, num_neuron]
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]

            # # using numpy
            # dist_a_list = np.linalg.norm(SA - self.SA_cache[int(label.cpu())], axis=1)
            # idx_a = np.argmin(dist_a_list, 0)

            dist_a_list = torch.linalg.norm(
                torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                    self.device), dim=1)
            idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    # # using numpy
                    # dist_b_list += np.linalg.norm(SA - self.SA_cache[j], axis=1).tolist()
                    dist_b_list += torch.linalg.norm(
                        torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                        dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            if (not np.isnan(dsa)) and (not np.isinf(dsa)):
                cove_set.add(int(dsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        return cove_set

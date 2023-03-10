from coverage.Coverage import *
import coverage.tool as tool


class CC(Coverage):
    '''
    Cluster-based Coverage, i.e., the coverage proposed by TensorFuzz
    '''
    def init_variable(self, hyper):
        assert hyper is not None
        self.threshold = hyper
        self.distant_dict = {}
        self.flann_dict = {}

        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.flann_dict[layer_name] = FLANN()
            self.distant_dict[layer_name] = []

    def update(self, dist_dict, delta=None):
        for layer_name in self.distant_dict.keys():
            self.distant_dict[layer_name] += dist_dict[layer_name]
            self.flann_dict[layer_name].build_index(np.array(self.distant_dict[layer_name]))
        if delta:
            self.current += delta
        else:
            self.current += self.coverage(dist_dict)

    def calculate(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        dist_dict = {}
        for (layer_name, layer_output) in layer_output_dict.items():
            dist_dict[layer_name] = []
            for single_output in layer_output:
                single_output = single_output.cpu().numpy()
                if len(self.distant_dict[layer_name]) > 0:
                    _, approx_distances = self.flann_dict[layer_name].nn_index(np.expand_dims(single_output, 0), num_neighbors=1)
                    exact_distances = [
                        np.sum(np.square(single_output - distant_vec))
                        for distant_vec in self.distant_dict[layer_name]
                    ]
                    buffer_distances = [
                        np.sum(np.square(single_output - buffer_vec))
                        for buffer_vec in dist_dict[layer_name]
                    ]
                    nearest_distance = min(exact_distances + approx_distances.tolist() + buffer_distances)
                    if nearest_distance > self.threshold:
                        dist_dict[layer_name].append(single_output)
                else:
                    self.flann_dict[layer_name].build_index(single_output)
                    self.distant_dict[layer_name].append(single_output)
        return dist_dict

    def coverage(self, dist_dict):
        total = 0
        for layer_name in dist_dict.keys():
            total += len(dist_dict[layer_name])
        return total

    def gain(self, dist_dict):
        increased = self.coverage(dist_dict)
        return

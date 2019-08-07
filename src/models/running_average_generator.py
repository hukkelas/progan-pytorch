from src.models.generator import Generator


class AverageGenerator(Generator):

    def __init__(self, start_channel_dim, image_channels, latent_size, beta):
        super().__init__(start_channel_dim, image_channels, latent_size)

        self.beta = beta

    def update(self, normal_generator):
        for avg_param, cur_param in zip(self.parameters(),
                                        normal_generator.parameters()):
            avg_param.data = self.beta * avg_param + (1-self.beta) * cur_param

    def extend(self, normal_generator):
        super().extend()
        for avg_param, cur_param in zip(self.new_parameters(),
                                        normal_generator.new_parameters()):
            assert avg_param.shape == cur_param.shape
            avg_param.data = cur_param.data


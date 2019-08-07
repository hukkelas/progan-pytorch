from src.models.generator import Generator


class AverageGenerator(Generator):

    def __init__(self, start_channel_dim, image_channels, latent_size, beta):
        super().__init__(start_channel_dim, image_channels, latent_size)

        self.beta = beta

    def update(self, normal_generator):
        for avg_param, cur_param in zip(self.parameters(),
                                        normal_generator.parameters()):
            avg_param.data = self.beta * avg_param + (1-self.beta) * cur_param

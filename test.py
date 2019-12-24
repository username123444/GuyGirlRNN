from trainer import *
import math
__trainer = Trainer("datasets/", ["girls", "guys"], clear_output=False)
#__trainer.train(epochs=100, dense_layers=[0], layer_sizes=[64], convolusion_layers=[3], batch_size=10, validation_split=0.3)
__g = round(__trainer.test("testguy.jpg")[0][0])
print(__g)
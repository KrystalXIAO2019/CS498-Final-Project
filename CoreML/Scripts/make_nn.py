import numpy as np
import coremltools
from coremltools.models.neural_network import NeuralNetworkBuilder, AdamParams, SgdParams

# Load the original Turi Create model and make it trainable.

model = coremltools.models.MLModel("FacesTuri.mlmodel")

# The original Turi Create model only has 3 output neurons, but we increment
# this to 10 neurons so that there is room to add new gestures. The weights
# for these new neurons are initially zeros.

spec = model._spec
layer = spec.neuralNetworkClassifier.layers[-2]

num_classes = 10
layer.innerProduct.outputChannels = num_classes

weights = np.zeros((num_classes - 3) * 1000)
biases = np.zeros(num_classes - 3)
labels = ["user" + str(i) for i in range(num_classes - 3)]

layer.innerProduct.weights.floatValue.extend(weights)
layer.innerProduct.bias.floatValue.extend(biases)
spec.neuralNetworkClassifier.stringClassLabels.vector.extend(labels)

# Make this model trainable.

builder = NeuralNetworkBuilder(spec=model._spec)

builder.make_updatable(["fullyconnected0"])
builder.set_categorical_cross_entropy_loss(name="lossLayer", input="labelProbability")
builder.set_epochs(10, [1, 10, 50])

# Using the SDG optimizer:
sgd_params = SgdParams(lr=0.001, batch=8, momentum=0)
sgd_params.set_batch(8, [1, 2, 8, 16])
builder.set_sgd_optimizer(sgd_params)

# Using the Adam optimizer:
# adam_params = AdamParams(lr=0.001, batch=8, beta1=0.9, beta2=0.999, eps=1e-8)
# adam_params.set_batch(8, [1, 2, 8, 16])
# builder.set_adam_optimizer(adam_params)

builder.spec.description.trainingInput[0].shortDescription = "Example image"
builder.spec.description.trainingInput[1].shortDescription = "True label"

coremltools.utils.save_spec(builder.spec, "FacesNN.mlmodel")

# Replace the weights of the last layer with random weights.

model = coremltools.models.MLModel("FacesNN.mlmodel")
print("Done!")

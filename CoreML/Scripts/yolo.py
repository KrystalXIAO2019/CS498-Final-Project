import coremltools
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.nearest_neighbors import KNearestNeighborsClassifierBuilder
import copy

base_model = coremltools.models.MLModel("../Models/TinyYOLO.mlmodel")
base_spec = base_model._spec

layers = copy.deepcopy(base_spec.neuralNetwork.layers)

print(layers[-1])

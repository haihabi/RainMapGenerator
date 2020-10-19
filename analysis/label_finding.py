from torchvision import transforms
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization

transform_training = transforms.Compose([
    transforms.ToTensor(),
    MaxNormalization(),
])
# TODO: add data augmentation
train_rds = RadarDataSet(training_data_pickle, transform=transform_training)

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from tensorflow.keras.models import load_model

from train import LAST_MODEL


model = load_model(LAST_MODEL)

model.summary()
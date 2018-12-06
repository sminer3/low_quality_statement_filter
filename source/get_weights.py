import numpy as np
import json
from keras.models import load_model

model = load_model('zero_value_model.h5')

layer1=np.vstack((np.transpose(model.get_weights()[1][:,None]),model.get_weights()[0])).tolist()
layer2=np.vstack((np.transpose(model.get_weights()[3][:,None]),model.get_weights()[2])).tolist()
layer3=np.vstack((np.transpose(model.get_weights()[5]),model.get_weights()[4])).flatten().tolist()
data={
 'id':'en',
 'data':{
     'layer1':layer1,
     'layer2':layer2,
     'output':layer3
 }
}

with open('weights.json', 'w') as outfile:
  json.dump(data, outfile)
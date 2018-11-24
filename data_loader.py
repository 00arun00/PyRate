def readGz(f):
    import gzip
    for l in gzip.open(f):
        yield eval(l)

def amazon_purchase_review():
    '''
    Loads the amazon purchase review data
    '''
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split as tts

    f_name=('Data/assignment1/train.json.gz')
    df = pd.DataFrame(readGz(f_name))[['itemID','reviewerID','rating']]

    data = df.values
    x = data[:,:2]
    y = data[:,2:]

    x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.5)

    return x_train,y_train,x_test,y_test

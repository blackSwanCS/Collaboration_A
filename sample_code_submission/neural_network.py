# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# class NeuralNetwork:
#     """
#     This class implements a neural network classifier with BatchNormalization and Dropout.
#     """

#     def __init__(self, train_data):
#         self.model = Sequential()

#         n_dim = train_data.shape[1]

#         self.model.add(Dense(64, input_dim=n_dim))
#         self.model.add(BatchNormalization())
#         self.model.add(Dense(64, activation="relu"))
#         self.model.add(Dropout(0.3))

#         self.model.add(Dense(32))
#         self.model.add(BatchNormalization())
#         self.model.add(Dense(32, activation="relu"))
#         self.model.add(Dropout(0.3))

#         self.model.add(Dense(16))
#         self.model.add(BatchNormalization())
#         self.model.add(Dense(16, activation="relu"))
#         self.model.add(Dropout(0.2))

#         self.model.add(Dense(1, activation="sigmoid"))  # Binary classification

#         self.model.compile(
#             loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
#         )

#         self.scaler = StandardScaler()

#     def fit(self, train_data, y_train, weights_train=None):
#         # Callbacks
#         early_stop = EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True
#         )

#         reduce_lr = ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=2,
#             min_lr=1e-6,
#             verbose=1
#         )

#         self.scaler.fit(train_data)
#         X_train = self.scaler.transform(train_data)

#         self.model.fit(
#             X_train,
#             y_train,
#             sample_weight=weights_train,
#             validation_split=0.2,
#             epochs=20,
#             batch_size=32,
#             verbose=2,
#             callbacks=[early_stop, reduce_lr]
#         )

#     def predict(self, test_data):
#         test_data = self.scaler.transform(test_data)
#         return self.model.predict(test_data).flatten().ravel()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from sklearn.isotonic import IsotonicRegression
import numpy as np
from tensorflow.keras.metrics import AUC



class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self, name, layers=[256, 128, 64], dropout_rate=0.2, 
                 epochs=30, batch_size=32, l2_reg=1e-4,input_dim=None, calibrate=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = name
        self.calibrate = calibrate
        self.scaler = StandardScaler()
        self.calibrator = None  # For isotonic regression

        # Build NN model
        self.model = Sequential()
        self.model.add(Dense(layers[0], input_dim=input_dim, activation="relu",
                             kernel_regularizer=l2(l2_reg)))
        self.model.add(BatchNormalization())
        # self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(layers[1], activation="relu", 
                             kernel_regularizer=l2(l2_reg)))
        self.model.add(BatchNormalization())
        # self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(layers[2], activation="relu", 
                             kernel_regularizer=l2(l2_reg)))
        self.model.add(BatchNormalization())
        # self.model.add(Dropout(dropout_rate))

        # Output layer
        self.model.add(Dense(1, activation="sigmoid"))

        # Compile model
        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy" , AUC(name="auc")]
        )

    def fit(self, train_data, y_train, weights_train=None, 
            val_data=None, y_val=None, weights_val=None):

        # Early stopping and LR reduction
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        self.scaler.fit(train_data)
        X_train = self.scaler.transform(train_data)
        
        if val_data is not None:
            # If user provided validation data → use it
            X_val = self.scaler.transform(val_data)
            history = self.model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                validation_data=(X_val, y_val, weights_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=2,
                callbacks=[early_stop, reduce_lr]
            )
        else:
            # If no validation data → hold out 20% automatically
            history = self.model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                validation_split=0.2,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=2,
                callbacks=[early_stop, reduce_lr]
            )
            
            
        self.history = history.history

        # Optional calibration step
        # if self.calibrate and val_data is not None:
        #     raw_preds = self.model.predict(X_val).ravel()
        #     self.calibrator = IsotonicRegression(out_of_bounds='clip')
        #     self.calibrator.fit(raw_preds, y_val, sample_weight=weights_val)
        #     print("Calibration step completed.")

    def predict(self, data):
        X_scaled = self.scaler.transform(data)
        raw_preds = self.model.predict(X_scaled).ravel()
        # if self.calibrate and self.calibrator is not None:
        #     return self.calibrator.predict(raw_preds)
        return raw_preds
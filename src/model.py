from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

class AttentionUNet:
    """Attention U-Net model for lung segmentation."""
    
    def __init__(self, img_rows=128, img_cols=128, df=64, uf=64):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_shape = (img_rows, img_cols, 1)
        self.df = df  # Downsampling filters
        self.uf = uf  # Upsampling filters

    def _conv2d(self, layer_input, filters, dropout_rate=0, bn=False):
        d = Conv2D(filters, (3, 3), padding='same')(layer_input)
        if bn:
            d = BatchNormalization()(d)
        d = Activation('relu')(d)
        d = Conv2D(filters, (3, 3), padding='same')(d)
        if bn:
            d = BatchNormalization()(d)
        d = Activation('relu')(d)
        if dropout_rate:
            d = Dropout(dropout_rate)(d)
        return d

    def _deconv2d(self, layer_input, filters, bn=False):
        u = UpSampling2D((2, 2))(layer_input)
        u = Conv2D(filters, (3, 3), padding='same')(u)
        if bn:
            u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    def _attention_block(self, F_g, F_l, F_int, bn=False):
        g = Conv2D(F_int, (1, 1), padding='valid')(F_g)
        if bn:
            g = BatchNormalization()(g)
        x = Conv2D(F_int, (1, 1), padding='valid')(F_l)
        if bn:
            x = BatchNormalization()(x)
        psi = Add()([g, x])
        psi = Activation('relu')(psi)
        psi = Conv2D(1, (1, 1), padding='valid')(psi)
        if bn:
            psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)
        return Multiply()([F_l, psi])

    def build_unet(self):
        """Build the Attention U-Net model."""
        inputs = Input(shape=self.img_shape)
        conv1 = self._conv2d(inputs, self.df)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = self._conv2d(pool1, self.df * 2, bn=True)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = self._conv2d(pool2, self.df * 4, bn=True)
        pool3 = MaxPooling2D((2, 2))(conv3)

        conv4 = self._conv2d(pool3, self.df * 8, dropout_rate=0.5, bn=True)
        pool4 = MaxPooling2D((2, 2))(conv4)

        conv5 = self._conv2d(pool4, self.df * 16, dropout_rate=0.5, bn=True)

        up6 = self._deconv2d(conv5, self.uf * 8, bn=True)
        conv6 = self._attention_block(up6, conv4, self.uf * 8, bn=True)
        up6 = Concatenate()([up6, conv6])
        conv6 = self._conv2d(up6, self.uf * 8)

        up7 = self._deconv2d(conv6, self.uf * 4, bn=True)
        conv7 = self._attention_block(up7, conv3, self.uf * 4, bn=True)
        up7 = Concatenate()([up7, conv7])
        conv7 = self._conv2d(up7, self.uf * 4)

        up8 = self._deconv2d(conv7, self.uf * 2, bn=True)
        conv8 = self._attention_block(up8, conv2, self.uf * 2, bn=True)
        up8 = Concatenate()([up8, conv8])
        conv8 = self._conv2d(up8, self.uf * 2)

        up9 = self._deconv2d(conv8, self.uf, bn=True)
        conv9 = self._attention_block(up9, conv1, self.uf, bn=True)
        up9 = Concatenate()([up9, conv9])
        conv9 = self._conv2d(up9, self.uf)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        return Model(inputs=inputs, outputs=outputs)

def compile_model(model, loss='binary_crossentropy', lr=1e-4):
    """Compile the U-Net model."""
    model.compile(loss=loss, optimizer=Adam(lr), metrics=['accuracy'])
    return model

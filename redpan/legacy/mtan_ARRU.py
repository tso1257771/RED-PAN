import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.keras.regularizers import l2


class unets:
    def __init__(
        self,
        input_size=(3000, 3),
        nb_filters=[6, 12, 18, 24, 30, 36],
        depth=6,
        kernel_size=7,
        kernel_init="he_uniform",
        kernel_regu=None,  # tf.keras.regularizers.l1(1e-6),
        bias_regu=None,  # tf.keras.regularizers.l1(1e-6),
        activation="relu",
        out_activation="softmax",
        dropout_rate=0.1,
        batchnorm=True,
        max_pool=False,
        pool_size=5,
        stride_size=5,
        upsize=5,
        padding="same",
        RRconv_time=3,
    ):

        self.input_size = input_size
        self.nb_filters = nb_filters
        self.depth = depth
        self.kernel_size = kernel_size
        self.kernel_init = kernel_init
        self.kernel_regu = kernel_regu
        self.bias_regu = bias_regu
        self.activation = activation
        self.out_activation = out_activation
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm
        self.max_pool = max_pool
        self.pool_size = pool_size
        self.stride_size = stride_size
        self.upsize = upsize
        self.padding = padding
        self.RRconv_time = RRconv_time

    def conv1d(self, nb_filter, stride_size=None):
        if stride_size:
            return Conv1D(
                nb_filter,
                self.kernel_size,
                padding=self.padding,
                strides=stride_size,
                bias_regularizer=self.bias_regu,
                kernel_initializer=self.kernel_init,
                kernel_regularizer=self.kernel_regu,
            )
        else:
            return Conv1D(
                nb_filter,
                self.kernel_size,
                padding=self.padding,
                bias_regularizer=self.bias_regu,
                kernel_initializer=self.kernel_init,
                kernel_regularizer=self.kernel_regu,
            )

    def mtan_att_block(
        self,
        pre_att_layer,
        pre_target,
        target,
        nb_filter_in,
        nb_filter_out,
        strides,
        mode=None,
        name=None,
    ):
        ## used in decoder, apply up-sampling
        if mode == "up":
            x = self.upconv_unit(
                pre_att_layer, nb_filter=nb_filter_out, concatenate_layer=pre_target
            )

        ## merge with layer at the same level of target layer before
        ## further convolution operations
        else:
            x = concatenate([pre_att_layer, pre_target])
        # attention layer
        x = Conv1D(nb_filter_in, 1, strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(nb_filter_in, 1, strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation("sigmoid")(x)
        ## element-wise multiplication with target layer
        ## (same as gating signal)
        if mode == "up":
            mul = Multiply(name=name)([x, target])

        ## down-sample/up-sample previous layer and then
        if mode == "down":
            mul = Multiply()([x, target])
            mul = self.conv_unit(
                mul, nb_filter=nb_filter_out, stride_size=strides, name=name
            )
        return mul

    def att_block(self, xl, gate, name=None):
        # xl = input feature (U net left hand side)
        # gate = gating signal (U net right hand side)
        F_l = int(xl.shape[1])
        F_g = int(gate.shape[1])
        F_int = int(xl.shape[2])

        W_x = Conv1D(
            F_l,
            F_int,
            strides=1,
            padding=self.padding,
            bias_regularizer=self.bias_regu,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.kernel_regu,
        )(xl)
        W_x_n = BatchNormalization()(W_x)

        W_g = Conv1D(
            F_g,
            F_int,
            strides=1,
            padding=self.padding,
            bias_regularizer=self.bias_regu,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.kernel_regu,
        )(gate)
        W_g_n = BatchNormalization()(W_g)

        add = Add()([W_x_n, W_g_n])
        add = Activation("relu")(add)

        psi = Conv1D(
            F_int,
            1,
            strides=1,
            padding=self.padding,
            bias_regularizer=self.bias_regu,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.kernel_regu,
        )(add)
        psi_n = BatchNormalization()(psi)
        psi_activate = Activation("sigmoid")(psi_n)

        mul = Multiply(name=name)([xl, psi_activate])

        return mul

    # when to add B.N. layer, before or after activation layer?
    def conv_unit(self, inputs, nb_filter, stride_size, name=None):

        if self.max_pool:
            u = self.conv1d(nb_filter, self.pool_size)(inputs)
            if self.batchnorm:
                u = BatchNormalization()(u)
            u = Activation(self.activation)(u)
            if self.dropout_rate:
                u = Dropout(self.dropout_rate)(u)
            u = MaxPooling1D(pool_size=self.pool_size, padding=self.padding, name=name)(
                u
            )
        else:
            if stride_size != None:
                u = self.conv1d(nb_filter, stride_size=stride_size)(inputs)
                if self.batchnorm:
                    u = BatchNormalization()(u)
                u = Activation(self.activation)(u)
                if self.dropout_rate:
                    u = Dropout(self.dropout_rate, name=name)(u)

            else:
                u = self.conv1d(nb_filter)(inputs)
                if self.batchnorm:
                    u = BatchNormalization()(u)
                u = Activation(self.activation)(u)
                if self.dropout_rate:
                    u = Dropout(self.dropout_rate, name=name)(u)
        return u

    def RRconv_unit(self, inputs, nb_filter, stride_size, name=None):

        if stride_size == None:
            u = self.conv_unit(inputs=inputs, nb_filter=nb_filter, stride_size=None)
        else:
            u = self.conv_unit(
                inputs=inputs, nb_filter=nb_filter, stride_size=stride_size
            )
        conv_1x1 = self.conv1d(nb_filter=nb_filter, stride_size=1)(u)
        for i in range(self.RRconv_time):
            if i == 0:
                r_u = u
            r_u = Add()([r_u, u])
            r_u = self.conv_unit(inputs=r_u, nb_filter=nb_filter, stride_size=None)

        return Add(name=name)([r_u, conv_1x1])

    def upconv_unit(
        self,
        inputs,
        nb_filter,
        concatenate_layer,
        apply_attention=False,
        att_transformer=False,
        name=None,
    ):
        # transposed convolution
        u = UpSampling1D(size=self.upsize)(inputs)
        u = self.conv1d(nb_filter, stride_size=None)(u)
        if self.batchnorm:
            u = BatchNormalization()(u)
        u = Activation(self.activation)(u)
        if self.dropout_rate:
            u = Dropout(self.dropout_rate)(u)
        # u.shape TensorShape([None, 128, 18])
        # concatenate_layer.shape TensorShape([None, 126, 18])
        shape_diff = u.shape[1] - concatenate_layer.shape[1]
        if shape_diff > 0:
            crop_shape = (shape_diff // 2, shape_diff - shape_diff // 2)
        else:
            crop_shape = None

        if apply_attention:
            if crop_shape:
                crop = Cropping1D(cropping=crop_shape)(u)
                att = self.att_block(xl=concatenate_layer, gate=crop)
                upconv = concatenate([att, crop], name=name)
            elif not crop_shape:
                att = self.att_block(xl=concatenate_layer, gate=u)
                upconv = concatenate([att, u], name=name)

        elif not apply_attention:
            if crop_shape:
                crop = Cropping1D(cropping=crop_shape)(u)
                upconv = concatenate([concatenate_layer, crop], name=name)
            elif not crop_shape:
                upconv = concatenate([concatenate_layer, u], name=name)

        return upconv

    def build_mtan_R2unet(
        self, pretrained_weights=None, input_size=None, nb_filters=None
    ):
        if nb_filters == None:
            nb_filters = self.nb_filters
        if input_size == None:
            input_size = self.input_size

        nb_filters = self.nb_filters[: self.depth]
        inputs = Input(input_size, name="input")

        # initial
        # ========== Encoder
        exp_Es = []
        Es = []
        PS_mtan_Es = []
        M_mtan_Es = []

        # initialize
        conv_init_exp = self.RRconv_unit(
            inputs=inputs, nb_filter=self.nb_filters[0], stride_size=None, name="E0"
        )
        PS_mtan_init = self.mtan_att_block(
            pre_att_layer=conv_init_exp,
            pre_target=conv_init_exp,
            target=conv_init_exp,
            nb_filter_in=self.nb_filters[0],
            nb_filter_out=self.nb_filters[0],
            mode="down",
            strides=None,
            name="PS_mtan_E0",
        )
        M_mtan_init = self.mtan_att_block(
            pre_att_layer=conv_init_exp,
            pre_target=conv_init_exp,
            target=conv_init_exp,
            nb_filter_in=self.nb_filters[0],
            nb_filter_out=self.nb_filters[0],
            mode="down",
            strides=None,
            name="M_mtan_E0",
        )

        Es.append(conv_init_exp)
        PS_mtan_Es.append(PS_mtan_init)
        M_mtan_Es.append(M_mtan_init)

        # Encoder
        for i in range(len(self.nb_filters) - 1):
            mtan_Eid = [f"P_mtan_E{i}", f"S_mtan_E{i}", f"M_mtan_E{i}"]
            if i == 0:
                exp_E = self.RRconv_unit(
                    inputs=conv_init_exp,
                    nb_filter=self.nb_filters[i],
                    stride_size=None,
                    name=f"exp_E{i}",
                )

                PS_mtan_E = self.mtan_att_block(
                    pre_att_layer=PS_mtan_init,
                    pre_target=conv_init_exp,
                    target=exp_E,
                    nb_filter_in=self.nb_filters[i],
                    nb_filter_out=self.nb_filters[i + 1],
                    mode="down",
                    strides=self.stride_size,
                    name=f"PS_mtan_E{i}",
                )
                M_mtan_E = self.mtan_att_block(
                    pre_att_layer=M_mtan_init,
                    pre_target=conv_init_exp,
                    target=exp_E,
                    nb_filter_in=self.nb_filters[i],
                    nb_filter_out=self.nb_filters[i + 1],
                    mode="down",
                    strides=self.stride_size,
                    name=f"M_mtan_E{i}",
                )

                E = self.RRconv_unit(
                    inputs=exp_E,
                    nb_filter=self.nb_filters[i + 1],
                    stride_size=self.stride_size,
                    name=f"E{i+1}",
                )
            else:
                exp_E = self.RRconv_unit(
                    inputs=E,
                    nb_filter=self.nb_filters[i],
                    stride_size=None,
                    name=f"exp_E{i}",
                )

                PS_mtan_E = self.mtan_att_block(
                    pre_att_layer=PS_mtan_E,
                    pre_target=E,
                    target=exp_E,
                    nb_filter_in=self.nb_filters[i],
                    nb_filter_out=self.nb_filters[i + 1],
                    mode="down",
                    strides=self.stride_size,
                    name=f"PS_mtan_E{i}",
                )
                M_mtan_E = self.mtan_att_block(
                    pre_att_layer=M_mtan_E,
                    pre_target=E,
                    target=exp_E,
                    nb_filter_in=self.nb_filters[i],
                    nb_filter_out=self.nb_filters[i + 1],
                    mode="down",
                    strides=self.stride_size,
                    name=f"M_mtan_E{i}",
                )

                E = self.RRconv_unit(
                    inputs=exp_E,
                    nb_filter=self.nb_filters[i + 1],
                    stride_size=self.stride_size,
                    name=f"E{i+1}",
                )

            PS_mtan_Es.append(PS_mtan_E)
            M_mtan_Es.append(M_mtan_E)
            Es.append(E)
            exp_Es.append(exp_E)

            # bottleneck layer
            if i == len(self.nb_filters) - 2:
                exp_E = self.RRconv_unit(
                    inputs=E,
                    nb_filter=self.nb_filters[i + 1],
                    stride_size=None,
                    name=f"exp_E{i+2}",
                )
                exp_Es.append(exp_E)

        # Decoder
        Ds = []
        PS_mtan_Ds = []
        M_mtan_Ds = []
        for i in range(len(self.nb_filters)):
            if i == 0:
                D = self.upconv_unit(
                    inputs=Es[-1],
                    nb_filter=self.nb_filters[-1 - i],
                    concatenate_layer=Es[-1 - i],
                    apply_attention=False,
                )
            else:
                D = self.upconv_unit(
                    inputs=D_fus,
                    nb_filter=self.nb_filters[-1 - i],
                    concatenate_layer=Es[-1 - i],
                    apply_attention=False,
                )

            D_fus = self.RRconv_unit(
                inputs=D,
                nb_filter=self.nb_filters[-1 - i],
                stride_size=None,
                name=f"D{i}",
            )

            PS_mtan_D = self.mtan_att_block(
                pre_att_layer=PS_mtan_Es[-1 - i],
                pre_target=D,
                target=D_fus,
                nb_filter_in=self.nb_filters[-1 - i],
                nb_filter_out=self.nb_filters[-1 - i],
                mode="up",
                strides=self.stride_size,
                name=f"PS_mtan_D{i}",
            )
            M_mtan_D = self.mtan_att_block(
                pre_att_layer=M_mtan_Es[-1 - i],
                pre_target=D,
                target=D_fus,
                nb_filter_in=self.nb_filters[-1 - i],
                nb_filter_out=self.nb_filters[-1 - i],
                mode="up",
                strides=self.stride_size,
                name=f"M_mtan_D{i}",
            )
            Ds.append(D_fus)
            PS_mtan_Ds.append(PS_mtan_D)
            M_mtan_Ds.append(M_mtan_D)

        ##========== Output map
        outPS = Conv1D(
            3,
            1,
            bias_regularizer=self.bias_regu,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.kernel_regu,
            name="pred_PS",
        )(PS_mtan_D)
        outM = Conv1D(
            2,
            1,
            bias_regularizer=self.bias_regu,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.kernel_regu,
            name="pred_mask",
        )(M_mtan_D)

        outPS_Act = Activation(self.out_activation, name="picker")(outPS)
        outM_Act = Activation(self.out_activation, name="detector")(outM)

        model = Model(inputs=inputs, outputs=[outPS_Act, outM_Act])

        # compile
        if pretrained_weights == None:
            return model

        else:
            model.load_weights(pretrained_weights)
            return model


if __name__ == "__main__":
    import tensorflow as tf

    args = {
        "input_size": (6000, 3),
        "nb_filters": [6, 12, 18, 24, 30, 36],
        "depth": 6,
        "kernel_size": 7,
        "kernel_init": "he_uniform",
        "kernel_regu": None,  # tf.keras.regularizers.l1(1e-6),
        "bias_regu": None,  # tf.keras.regularizers.l1(1e-6),
        "activation": "relu",
        "out_activation": "softmax",
        "dropout_rate": 0.1,
        "batchnorm": True,
        "max_pool": False,
        "pool_size": 5,
        "stride_size": 5,
        "upsize": 5,
        "padding": "same",
        "RRconv_time": 3,
    }
    frame = unets(**args)
    arru = frame.build_mtan_R2unet()
    print(arru.summary())

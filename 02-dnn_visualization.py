#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import keras
from keras.preprocessing import image
from keras.utils import plot_model


def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    # 学習済ファイルの確認
    if len(sys.argv)==1:
        print('使用法: python dnn_visualization.py 学習済ファイル名.h5')
        sys.exit()
    savefile = sys.argv[1]

    # モデルのロード
    model = keras.models.load_model(savefile)

    # モデル名
    # print("モデル名")
    print(model.get_layer)

    # モデルの概要を表示
    model.summary()

    # モデルの概要図をファイル出力
    plot_model(model, to_file='model.png', show_shapes=True)


if __name__ == '__main__':
    main()

文字画像データセット(平仮名版)
====================

画像形式はJPGで、計3万画像あります。

<!-- 
### データセットの権利

「PDM（パブリック・ドメイン・マーク）」&lt; https://creativecommons.org/publicdomain/mark/1.0/deed.ja &gt; -->


こちらからダウンロードできます(どの形式でも内容は同じです)。

* [文字画像データセット(平仮名版) (zip形式)](https://www.hi.u-tokyo.ac.jp/cdps/datadriven/dataset/hiragana.zip)　(約45MB)


内訳は次のとおりです。

（計）27,743

文字 | ディレクトリ |   画像数 | 文字 | ディレクトリ |   画像数 | 文字 | ディレクトリ |   画像数 | 文字  | ディレクトリ |    画像数
:--:|:------:| -----:|:--:|:------:| -----:|:--:|:------:| -----:|:---:|:------:| ------:
あ | U3042 | 372 | せ | U305B | 280 | は | U306F | 299 | よ | U3088 | 616
い | U3044 | 758 | そ | U305D | 293 | ひ | U3072 | 258 | ら | U3089 | 489
う | U3046 | 840 | た | U305F | 1,054 | ふ | U3075 | 328 | り | U308A | 919
え | U3048 | 86 | ち | U3061 | 389 | へ | U3078 | 1,345 | る | U308B | 554
お | U304A | 314 | つ | U3064 | 809 | ほ | U307B | 199 | れ | U308C | 310
か | U304B | 1,322 | て | U3066 | 1,512 | ま | U307E | 581 | ろ | U308D | 239
き | U304D | 724 | と | U3068 | 1,357 | み | U307F | 93 | わ | U308F | 391
く | U304F | 922 | な | U306A | 630 | む | U3080 | 101 | ゑ | U3091 | 194
け | U3051 | 284 | に | U306B | 1,017 | め | U3081 | 226 | を | U3092 | 671
こ | U3053 | 380 | ぬ | U306C | 93 | も | U3082 | 1,007 | ん | U3093 | 739
さ | U3055 | 616 | ね | U306D | 143 | や | U3084 | 518 | 
し | U3057 | 1,497 | の | U306E | 1,726 | ゆ | U3086 | 248 | 

データセットの活用例として、機械学習による自動分類プログラムを試作しました。オープンソースの深層学習フレームワーク[Chainer](http://chainer.org/)のサンプルプログラムを改変したものです。Chainerの実行環境の他、Pillowパッケージが必要です。

https://colab.research.google.com/drive/1n6roqxdqSxVWErOUK2p9icb0kX39wOVB?usp=sharing

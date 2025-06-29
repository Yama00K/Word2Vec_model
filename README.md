# Word2Vec model

## このモデルの目的

このモデルは、文章データから各単語の前後関係を基に単語の分散表現を生成することを目的としている。

## 環境構築

本モデルはPyTorchを用いて実装しているため、動作環境によってはGPUを用いて動作させることができる。以下に私が利用している動作環境ごとの構築について示す。

### venv(Mac環境で利用)
```bash
#　仮想環境内でrequirements_CPU.txtをインストール

python3.12 -m venv myenv

pip install -r requirements_CPU.txt
```
上記のコマンドを用いて仮想環境を作成し、requirements_CPU.txtをインストールして環境を構築する。
### conda(Windows環境で利用)
```bash
# 仮想環境内でrequirements_GPU.txtをインストール

conda create -n myenv python=3.12

pip install -r requirments_GPU.txt
```
上記のコマンドを用いて仮想環境を作成し、requirements_GPU.txtをインストールして環境を構築する。
また、PyTorchのバージョンに対応するNVIDIAのドライバをインストールする必要がある。
私の環境では[NVIDIA Driver Version: 573.06]をインストールしている。
利用するNVIDIAのドライバのバージョンによっては、インストールするPyTorchのバージョンをcuda toolkitのバージョンに合わせる必要がある

## モデルのパラメータ

モデルのパラメータはmain.pyにて変更できる。
```python
num_workers = 8         # データの受け渡し部分の数
embedding_dim = 100     # ベクトルの次元数(中間層のノード数)
batch_size = 100        # 1バッチあたりのデータ数
context_size = 5        # targetの両隣のコンテキスト数
num_negative = 5        # 学習次に用いる間違いデータの数
learning_rate = 0.001   # 学習率
epochs = 50             #エポック数
```
## モデルの動作

モデルを動作させるにはmain.pyを動作させる。
```python
python3 main.py
```

## 出力されたベクトルの確認

logs/CBOW_model下に動作させた時の日付に基づいたディレクトリが生成され、そこへ動作結果が保存される。出力したベクトルはwordvecs.pklに保存されている。pklファイルにdict型で保存されているので、データを取り出す際は、以下に示すdictに合わせて取り出す。
```python
save_data ={
'wordvecs_in': wordvecs_in,             # 入力層側のベクトル
'wordvecs_out': wordvecs_out,           # 出力層側のベクトル
'word_to_id': datamodule.word_to_id,    # 単語からidへの変換dict
'id_to_word': datamodule.id_to_word     # idから単語への変換dict
}
```
また、logs/CBOW_model下にtensorboardを使ってlogを保存しているため、学習途中のlossはtensorboardを使って確認できる。

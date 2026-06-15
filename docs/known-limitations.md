# 既知の制約

この文書は、現在の EUB Changes で確認している実用上の制約をまとめます。

安全上の注意は README の Safety セクションを参照してください。

## Digitone II への送信

EUB Changes は Digitone II 向けに開発・実機確認されています。

SysEx送信は選択したMIDIポート上のDigitone IIへpattern dataを書き込むため、初回は空Projectまたはバックアップ済みProjectで確認してください。

## iReal Pro import

- iReal import は、同梱の `ireal-musicxml` で iReal data を MusicXML へ変換してから取り込みます。
- iReal Proの表示レイアウト、コメント、alternate chord、backing track情報が完全に再現されるとは限りません。
- MIDI生成（`musicxml-midi`）は同梱していません。
- converter warning が取得できた場合は import warning として表示します。

## 公開プレイリストのインポート

- Import セクションのプルダウンから公開 iReal Pro playlist を選ぶ場合、インポート時にネットワーク接続が必要です。
- オフライン環境では、iReal Pro の `.html` ファイルを手動で用意して file upload からインポートしてください。
- 大規模playlistは変換に時間がかかる場合があります。

## MIDI環境

- Preview / Send を使うには、利用環境でMIDI backendが利用可能である必要があります。
- EUB Changes はMIDIポートを自動選択しません。送信前に必ず送信先ポートを確認してください。

## LPC Lead Layer (beta)

LPC Lead Layer は v0.3.0 の beta feature です。Tracks 9–16 を占有し、外部 MIDI 入力をリアルタイムに処理しません。Track Level の調整は Digitone II 側またはユーザーの MIDI 環境で行ってください。

## 対応範囲

EUB Changes は Digitone II 向けのmachine-live支援ツールです。

他のElektron機器、他社MIDI機器、またはDigitone II以外のSysEx互換性は保証しません。
from speechbrain.inference.separation import SepformerSeparation
import soundfile as sf

model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-libri3mix",
    savedir="pretrained_sepformer_libri3mix",
    run_opts={"device": "cuda:0"},
)

est_sources = model.separate_file("arabic_data/mix/sample_0000.wav")

# est_sources is typically [batch, time, speakers]
pred1 = est_sources[0, :, 0].detach().cpu().numpy()
pred2 = est_sources[0, :, 1].detach().cpu().numpy()
pred3 = est_sources[0, :, 2].detach().cpu().numpy()

sf.write("pred1.wav", pred1, 8000)
sf.write("pred2.wav", pred2, 8000)
sf.write("pred3.wav", pred3, 8000)

print("Saved pred1.wav, pred2.wav, pred3.wav")
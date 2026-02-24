import sys
import re

def modify_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Add Wavelet to imports
    if 'Wavelet' not in content:
        content = content.replace(
            'timestep_embedding,',
            'timestep_embedding,\n    Wavelet,'
        )

    # 2. ResBlock.__init__
    content = content.replace(
        'down=False,\n    ):',
        'down=False,\n        use_wavelet=False,\n    ):'
    )
    content = content.replace(
        'self.use_scale_shift_norm = use_scale_shift_norm\n\n        self.in_layers = nn.Sequential(\n            normalization(channels),\n            nn.SiLU(),',
        'self.use_scale_shift_norm = use_scale_shift_norm\n        Activation = Wavelet if use_wavelet else nn.SiLU\n\n        self.in_layers = nn.Sequential(\n            normalization(channels),\n            Activation(),'
    )
    content = content.replace(
        'self.emb_layers = nn.Sequential(\n            nn.SiLU(),',
        'self.emb_layers = nn.Sequential(\n            Activation(),'
    )
    content = content.replace(
        'self.out_layers = nn.Sequential(\n            normalization(self.out_channels),\n            nn.SiLU(),',
        'self.out_layers = nn.Sequential(\n            normalization(self.out_channels),\n            Activation(),'
    )

    # 3. UNetModel.__init__
    content = content.replace(
        'legacy=True,\n    ):',
        'legacy=True,\n        use_wavelet=False,\n    ):'
    )
    content = content.replace(
        'self.predict_codebook_ids = n_embed is not None\n\n        time_embed_dim = model_channels * 4\n        self.time_embed = nn.Sequential(\n            linear(model_channels, time_embed_dim),\n            nn.SiLU(),',
        'self.predict_codebook_ids = n_embed is not None\n\n        Activation = Wavelet if use_wavelet else nn.SiLU\n\n        time_embed_dim = model_channels * 4\n        self.time_embed = nn.Sequential(\n            linear(model_channels, time_embed_dim),\n            Activation(),'
    )
    content = content.replace(
        'self.out = nn.Sequential(\n            normalization(ch),\n            nn.SiLU(),',
        'self.out = nn.Sequential(\n            normalization(ch),\n            Activation(),'
    )

    # 4. EncoderUNetModel.__init__
    content = content.replace(
        'pool="adaptive",\n        *args,\n        **kwargs\n    ):',
        'pool="adaptive",\n        use_wavelet=False,\n        *args,\n        **kwargs\n    ):'
    )
    content = content.replace(
        'self.num_heads_upsample = num_heads_upsample\n\n        time_embed_dim = model_channels * 4\n        self.time_embed = nn.Sequential(\n            linear(model_channels, time_embed_dim),\n            nn.SiLU(),',
        'self.num_heads_upsample = num_heads_upsample\n\n        Activation = Wavelet if use_wavelet else nn.SiLU\n\n        time_embed_dim = model_channels * 4\n        self.time_embed = nn.Sequential(\n            linear(model_channels, time_embed_dim),\n            Activation(),'
    )
    content = content.replace(
        'self.out = nn.Sequential(\n            normalization(ch),\n            nn.SiLU(),',
        'self.out = nn.Sequential(\n            normalization(ch),\n            Activation(),'
    )

    # 5. Add use_wavelet to ResBlock calls
    content = content.replace(
        'use_scale_shift_norm=use_scale_shift_norm,',
        'use_scale_shift_norm=use_scale_shift_norm,\n                        use_wavelet=use_wavelet,'
    )
    
    # 6. EncoderUNetModel out block has two nn.SiLU() calls, we should probably handle them just in case. They are in blocks like `nn.Sequential(..., nn.SiLU(), ...)`?
    # Actually EncoderUNetModel also ends in `out = nn.Sequential(...)` which we covered above but wait, my sed script above for EncoderUNetModel replaced `nn.SiLU()`. But I should also check if `nn.SiLU()` was correctly replaced.
    # Let's save.
    with open(file_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    modify_file(sys.argv[1])

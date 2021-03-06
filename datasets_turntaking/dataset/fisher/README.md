# Fisher Dialog Dataset

* Paper: [The Fisher Corpus: a Resource for the Next Generations of Speech-to-Text](https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/lrec2004-fisher-corpus.pdf)
* [Fisher Audio, LDC](https://catalog.ldc.upenn.edu/LDC2004S13)
  - Download fisher and extract files
  - 001: `fisher_eng_tr_sp_LDC2004S13.zip.001`
  - 002: `fisher_eng_tr_sp_LDC2004S13.zip.002`
  - Combine the two zips
    - `cat fisher_eng_tr_sp_LDC2004S13.zip.00* > fisher.zip`
  - Unzip: `unzip fisher.zip`
* [Fisher Transcript, LDC](https://catalog.ldc.upenn.edu/LDC2004T19)
  - Download transcripts
  - Extract: `tar zxvf fe_03_p1_tran_LDC2004T19.tgz`


```bash
Root/
│   # TRANSCRIPTS
├── fe_03_p1_tran
│   ├── data
│   ├── doc
│   └── index.html
│
│   # AUDIO
├── fisher_eng_tr_sp_d1 # Audio
│   ├── 0readme.txt
│   ├── audio
│   ├── fe_03_p1_sph1
│   ├── filetable.txt
│   └── volume_id.txt
├── ...
│
└── fisher_eng_tr_sp_d7
    ├── 0readme.txt
    ├── audio
    ├── fe_03_p1_sph7
    ├── filetable.txt
    └── volume_id.txt
```

## Preprocess

The audio files requires preprocessing (`.sph`-> `.wav`).

* Download [sph2pipe_v2.5](https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools)
  - Extract: `tar -xf sph2pipe_v2.5.tar.gz`
  - CD into folder
  - Compile: `gcc -o sph2pipe *.c -lm` 
    - creates `sph2pipe`
  - Then make sure that the file is located somewhere in your `$PATH` so that it can be used by python
* Run: `python datasets_turntaking/fisher/preprocess --root $PATH_TO_FISHER_DIR`
  - This will convert sph to wav (same filename different ext) and delete the original sph files.

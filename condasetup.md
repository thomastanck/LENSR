First activate conda, which has been helpfully provided by other people

```
source /temp/miniconda/bin/activate
```

(if it isn't there, figure out how to activate lol)

Try to create the conda env as described in the README.
If it doesn't work, you may need to destroy all versioning and unfreeze the entire thing...
That means removing everything after the '=' sign in environment.yml

After that, the env should be created, but we still need to install some additional packages:

```
conda activate LENSR
pip install python-sat
```

python-sat in particular isn't in the conda repositories, so we have to get it from pip.
Ensure that you're running these commands in the LENSR environment!



import celltypist
from celltypist import models

#Download all the available models.
models.download_models()
#Update all models by re-downloading the latest versions if you think they may be outdated.
models.download_models(force_update = True)

models.models_description(on_the_fly = True)
# CivitaiBatchHelper
### If you used an older version, output folders now follow the same structure as the models (nested folder) 
---
 
When you run it for the first time, it will ask you for an civitai API key and a path to your models folder.  
Then it will scan all of your checkpoints/loras ect... and create a .txt file with
- Creator/uploader
- Model type
- Model name
- SD Base version
- Version name
- Civitai link
- Trigger Words (if available)
- Model description
- Example Prompts (if available)
- Training Data (if available)
  - Clip Skip
  - Epoch
  - Steps
  - Batch

and also download the preview images with embeded pngInfo

---
Civitai API key can be created at the bottom of https://civitai.com/user/account  
You can change the API key/model and output directory in the settings.json  
Required module will be installed if not present: PIL(pillow)/aiohttp  

---
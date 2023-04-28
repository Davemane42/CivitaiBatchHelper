# CivitaiBatchHelper
 
When you run it for the first time, it will ask you for an civitai API key and a path to your models folder.  
Then it will scan all of your checkpoints/loras ect... and create a .txt file with
- Creator/uploader
- Model type
- Model name
- Version name
- Civitai link
- Model description
- Version description (if available)
- Trigger Words (if available)
- Example Prompts (if available)

and also download the preview images with embeded pngInfo

---
Civitai API key can be created at the bottom of https://civitai.com/user/account  
You can change the API key/model and output directory in the settings.json  
Required module will be installed if not present: PIL(pillow)/aiohttp  

---
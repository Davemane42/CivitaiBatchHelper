import hashlib
import os
import requests
import json
import glob
import re
import shutil

import sys
import subprocess
import importlib.util

import asyncio
import time

from tqdm import tqdm

exifRename = {
    'negativePrompt': 'Negative prompt',
    'cfgScale': 'CFG scale',
    'ENSD': 'ENSD'
}
exifOrder = ['Prompt', 'Negative prompt', 'Steps', 'Sampler', 'CFG scale', 'Seed', 'Size', 'Model hash', 'Model', 'Denoising strength', 'Clip skip', 'ENSD', 'Hires upscale', 'Hires steps', 'Hires upscaler']

base_url = 'https://civitai.com/api/v1'
user_agent = 'CivitaiLink:CivitaiBatchHelper'
download_chunk_size = 8192

def is_installed(package, package_overwrite=None):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        pass

    package = package_overwrite or package

    if spec is None:
        print(f"Installing {package}...")
        command = f'"{sys.executable}" -m pip install {package}'
  
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ)

        if result.returncode != 0:
            print(f"Couldn't install\nCommand: {command}\nError code: {result.returncode}")


is_installed("aiohttp")
is_installed("PIL")

import aiohttp
from io import BytesIO
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Get and download preview images
async def get_all_images(images):
    with tqdm(total=len(images), unit='images') as bar:
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[get_image(session, image, bar) for image in images])

async def get_image(session, image, bar):
    headers = {'User-Agent': user_agent, 'Authorization': f"Bearer {settings['civitai_api_key']}"}
    try:
        async with session.get(url=image['url'], headers=headers) as response:
            
            image_data = BytesIO()

            async for chunk in response.content.iter_chunked(download_chunk_size):
                image_data.write(chunk)

            if response.status != 200:
                print(f"Scraping {image['path']} failed due to the return code {response.status}")
                return

            pil_image = Image.open(image_data)
            metadata = PngInfo()
            if image['pngInfo'] != '':
                metadata.add_text('parameters', image['pngInfo'])
                
            pil_image.save(image['path'], format='PNG', pnginfo=metadata, compress_level=4)

            bar.update(1)

    except Exception as e:
        print(f"Unable to get url {e.__class__}.")
        print('\n'.join(e.args))

# Get individual model description
async def get_all_models(models):
    result = []
    with tqdm(total=len(models), unit='models') as bar:
        async with aiohttp.ClientSession() as session:
            result.extend(await asyncio.gather(*[get_model(session, hashKey, modelID, bar) for hashKey, modelID in models.items()]))

    return result

async def get_model(session, hashKey, modelID, bar):
    url = f'{base_url}/models/{modelID}'
    headers = {'User-Agent': user_agent, 'Authorization': f"Bearer {settings['civitai_api_key']}"}
    try:
        async with session.get(url=url, headers=headers) as response:
            resp = await response.json()
            bar.update(1)
            return [hashKey, resp]
    except Exception as e:
        print(f'Unable to get "{url}" due to.')
        print('\n'.join(e.args))

#
async def get_all_models_by_hash(hashes):
    results = []
    batches = []
    for i in range(0, len(hashes), 100):
        batches.append(json.dumps(hashes[i:i + 100]))

    try:
        async with aiohttp.ClientSession() as session:
            result = (await asyncio.gather(*[get_model_by_hash(session, batch) for batch in batches]))
            for x in result:
                results.extend(x)
    except Exception as e:
        print(f"Unable to get batch of hash due to.")
        print('\n'.join(e.args))
    
    return results

async def get_model_by_hash(session, hashes):
    headers = {'User-Agent': user_agent, 'Authorization': f"Bearer {settings['civitai_api_key']}"}
    headers['Content-Type'] = 'application/json'

    async with session.post(url=f'{base_url}/model-versions/by-hash', headers=headers, data=hashes) as response:
        return await response.json()

def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

def save_cache():
    with open(cachePath, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)

def get_relative_path(path):
    return os.path.relpath(path, start=settings['models_path'])

def lazyHTML2Text(text):
    text = re.sub(r'<a.*?href="(.*?)".*?<\/a>', r'\g<1>', text) # Link
    text = re.sub(r'<img.*?src="(.*?)".*?>', r'\g<1>', text) # Image
    text = re.sub(r'<li>(.*?)<\/li>', r'- \g<1>', text) # List item

    text = re.sub(r'<div(.*?)\/div>', r'\n\g<1>\n', text) # div
    text = re.sub(r'<h[0-9]>(.*?)<\/h[0-9]>', r'\g<1>\n', text) # Header
    text = re.sub(r'<p>(.*?)<\/p>', r'\g<1>\n', text) # Paragraph
    text = re.sub(r'<br[^>]*>', '\n', text) # br

    text = re.sub(r'<[^>]*>', '', text) # Remove all other <tag> or </tag>

    return '\n'.join([f'\t{line.strip()}' for line in text.splitlines()])

# Settings
folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(folderPath, exist_ok=True)

defaultSettings = {'civitai_api_key': None, 'models_path': None, 'output_path': None, 'full_description': False}
settingPath = os.path.join(folderPath, 'settings.json')
try:
    with open(settingPath, 'r') as f:
        settings = json.load(f)
except FileNotFoundError:
    with open(settingPath, 'w') as f:
        json.dump(defaultSettings, f, ensure_ascii=False, indent=1)
        settings = defaultSettings

settings = defaultSettings | settings

def changeValue(key, text):
    while True:
        inputText = input(text)
        settings[key] = inputText
            
        inputText = input("Confirm? (y/n): ")
        if inputText.lower() == "y" or inputText == "":
            break
        elif inputText.lower() != "n":
            print("Input a valid value")

# Api Key
if not settings['civitai_api_key']:
    print('Civitai API key can be generated at the bottom of your "Account settings" page')
    changeValue('civitai_api_key', 'API Key: ')

# Input folder
if not settings['models_path']:
    print('\nThe models folder is where all your checkpoints/loras are located')
    while True:
        changeValue('models_path', 'Model folder path: ')

        if os.path.exists(settings['models_path']):
            if os.path.isfile(settings['models_path']):
                settings['models_path'] = os.path.dirname(settings['models_path'])
            break
        else:
            print("Input a valid input path")

# Output folder
if not settings['output_path']:
    print('\nThe output folder is where all the data will be generated')
    while True:
        print(f'Default output path is: "{folderPath}"')

        inputText = input("Confirm? (y/n): ")
        if inputText.lower() == "y" or inputText == "":
            settings['output_path'] = folderPath
            break
        elif inputText.lower() == "n":
            break
        else:
            print("Input a valid value")
    if not settings['output_path']:
        while True:
            changeValue('output_path', 'Output folder path: ')

            if os.path.exists(settings['output_path']):
                if os.path.isfile(settings['output_path']):
                    settings['output_path'] = os.path.dirname(settings['output_path'])
                break
            else:
                print("Input a valid input path")

# Save Settings
with open(settingPath, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=1)

# Cache
cachePath = os.path.join(folderPath, settings['output_path'], 'cache.json')
try:
    with open(cachePath, 'r') as f:
        cache = json.load(f)
except FileNotFoundError:
    with open(settingPath, 'w') as f:
        json.dump({}, f, ensure_ascii=False, indent=1)
        cache = {}

files = []
filetypes = ('*.safetensors', '*.ckpt')
for ext in filetypes:
    files.extend(glob.glob(os.path.join(settings['models_path'], '**', ext), recursive=True))

fileCount = len(files)
lenCount = len(str(fileCount))

if fileCount == 0:
    print(f'No files detected:')
    input('press "enter" to exit')
    exit()
else:
    print(f"\nfound: {fileCount} models\n")

data = {}
for i, file in enumerate(files):
    
    fileName = os.path.split(file)[1]
   
    if fileName in cache:
        sha256 = cache[fileName]['SHA256']
    else:
        fileSize = os.stat(file).st_size / (1024*1024*1024) # in GB

        if fileSize > 1:
            print(f'\nFile size is {fileSize:.2f}GB, this might take a while')

        print(f'{i+1:>{lenCount}d}/{fileCount} caching hash "{fileName}" -> ',end="")
        sha256 = calculate_sha256(file).upper()
        cache[fileName] = {}
        cache[fileName]['SHA256'] = sha256
        cache[fileName]['type'] = None
        print(f'"{sha256}"')

        if fileSize > 1:
            save_cache()
    
    if sha256 in data:
        print(f'"{get_relative_path(file)}" is a duplicate of "{get_relative_path(data[sha256]["path"])}"')
        continue
    
    data[sha256] = {}
    data[sha256]['path'] = file
    data[sha256]['fileName'] = fileName
    data[sha256]['version'] = {}
    

save_cache()

# civitai request by hash
startTime = time.time()

hashes = [value for value in data.keys()]
print(f"\nRequesting: {len(hashes)} models from Civitai.com")
results = asyncio.run(get_all_models_by_hash(hashes))

print(f"Agregating data of {len(results)} matched models:")
models = {}
for result in results:
    found = False
    for file in result['files']:
        
        if not 'SHA256' in file['hashes']:
            continue

        sha256 = file['hashes']['SHA256']

        if not sha256 in data:
            continue

        if file['type'] == 'VAE':
            continue
        
        data[sha256]['version'] = result
        models[sha256] = result['modelId']
        found = True
        break

    if not found:
        print(f"ERROR: {result['model']['name']}")

# get model data
result = asyncio.run(get_all_models(models))
for x in result:
    data[x[0]]['model'] = x[1]

print(f'Took {time.time() - startTime:.2f} seconds\n')

# Main loop
dataCount = len(data)
lenCount = len(str(dataCount))
for i, hashKey in enumerate(data):
    model = data[hashKey]

    counter = f'{i+1:>{lenCount}d}/{dataCount}'

    if model['version'] == {}:
        path = get_relative_path(model['path'])
        print(f'{counter} Could not find Civitai Data for "{path}"')
        continue
    
    fileName = os.path.splitext(model['fileName'])[0]
    
    modelName = model['version']['model']['name']
    modelType = model['version']['model']['type']
    modelID = model['version']['modelId']

    versionName = model['version']['name']
    versionID = model['version']['id']
    
    newPath = os.path.join(folderPath, settings['output_path'], modelType, fileName)

    os.makedirs(os.path.join(folderPath, settings['output_path'], modelType), exist_ok=True)
    os.makedirs(newPath, exist_ok=True)
    cache[model['fileName']]['type'] = modelType

    creator = model['model']['creator']['username']
    fileText = f"Creator: {creator}\nhttps://civitai.com/user/{creator}/models"

    fileText += f'\n\nModel type:   {modelType}\nModel Name:   {modelName}\nVersion Name: {versionName}'
    fileText += f'\nhttps://civitai.com/models/{modelID}?modelVersionId={versionID}'

    fileText += f"\n\nAllow no credit: {model['model']['allowNoCredit']}"
    fileText += f"\nAllow Comercial Use: {model['model']['allowCommercialUse']}"
    fileText += f"\nAllow Derivative: {model['model']['allowDerivatives']}"
    fileText += f"\nAllow Different License: {model['model']['allowDifferentLicense']}"

    

    if model['version']['description']:
        fileText += f"\n\nVersion Description:\n{lazyHTML2Text(model['version']['description'])}"

    if model['model']['description']:
        fileText += f"\n\nModel Description:\n{lazyHTML2Text(model['model']['description'])}"

    if len(model['version']['trainedWords']) != 0:
        fileText += f"\n\nTrigger Words:\n{', '.join(model['version']['trainedWords'])}"

    # info txt file
    newImages = []
    if len(model['version']['images']) != 0:
        prompts = ''
        for image in model['version']['images']:
            
            imageUrl = image['url']

            imageID = os.path.splitext(os.path.basename(imageUrl))[0]
            imagePath = os.path.join(newPath, f'preview_{imageID}.png')
            
            # reconstruct auto1111 exif data
            pngInfo = ''
            if image['meta']:
                meta = {}
                for key, parameter in image['meta'].items():
                    if key in exifRename:
                        key = exifRename[key]
                    else:
                        key = key.capitalize()

                    if key in exifOrder:
                        meta[key] = parameter
                
                parameters = []
                for key in exifOrder:

                    if key in meta:
                        v = meta[key]

                        if key == 'Prompt' and 'Prompt' in meta:
                            pngInfo += f"{meta['Prompt']}\n"

                        elif key == 'Negative prompt' and 'Negative prompt' in meta:
                            pngInfo += f"Negative prompt: {meta['Negative prompt']}\n"

                        else:
                            parameters.append(f'{key}: {v}')

                pngInfo += ', '.join(parameters)

            # Remove blank lines
            pngInfo = '\n'.join([ll.rstrip() for ll in pngInfo.splitlines() if ll.strip()])

            if not os.path.exists(imagePath):
                newImages.append({'url': imageUrl, 'path': imagePath, 'pngInfo': pngInfo})
            
            prompts += f'\n\n{pngInfo}'

        if prompts != '':
            fileText += f'\n\nExample Prompts:{prompts}'

    with open(os.path.join(newPath, f'{fileName}.txt'), 'w', encoding='utf-8') as f:
        f.write(fileText)

    # Download images preview files
    if len(newImages) != 0:
        print(f'\n{counter} Found {modelType} "{modelName}" version "{versionName}" on Civitai with {len(newImages)} preview images')
        asyncio.run(get_all_images(newImages))

# Remove old folder/cache
print()
for file, item in cache.items():
    if item['SHA256'] in data:
        continue
    if not item['type']:
        continue
    path = os.path.join(folderPath, settings['output_path'], item['type'], os.path.splitext(file)[0])
    if os.path.exists(path):
        print(f'"{file}" has been removed by user, deleting "{item["type"]}\{os.path.splitext(file)[0]}"')
        shutil.rmtree(path)

save_cache()
input('\nDone (press "enter" to exit)')
import hashlib
import os
import requests
import json
import glob
import re
import shutil

from io import BytesIO
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tqdm import tqdm

exifRename = {
    'negativePrompt': 'Negative prompt',
    'cfgScale': 'CFG scale',
    'ENSD': 'ENSD'
}
exifOrder = ['Prompt', 'Negative prompt', 'Steps', 'Sampler', 'CFG scale', 'Seed', 'Size', 'Model hash', 'Model', 'Denoising strength', 'Clip skip', 'ENSD', 'Hires upscale', 'Hires steps', 'Hires upscaler']

# Copied req function from https://github.com/civitai/sd_civitai_extension/blob/c4d4b2e374eccb5f192929a1332e46852f494173/civitai/lib.py#L69

base_url = 'https://civitai.com/api/v1'
user_agent = 'CivitaiLink:ComfyUI'
download_chunk_size = 8192

def req(endpoint, method='GET', data=None, params=None, headers=None):
    """Make a request to the Civitai API."""
    if headers is None:
        headers = {}
    headers['User-Agent'] = user_agent
    api_key = settings['civitai_api_key']
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'
    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data)
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
    if params is None:
        params = {}
    response = requests.request(method, base_url+endpoint, data=data, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f'Error: {response.status_code} {response.text}')
    return response.json()


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

# Settings
#defaultInputPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(folderPath, exist_ok=True)

defaultSettings = {'civitai_api_key': None, 'models_path': None}
settingPath = os.path.join(folderPath, 'settings.json')
try:
    with open(settingPath, 'r') as f:
        settings = json.load(f)
except FileNotFoundError:
    with open(settingPath, 'w') as f:
        json.dump(defaultSettings, f, ensure_ascii=False, indent=1)
        settings = defaultSettings

settings = defaultSettings | settings

# Cache
cachePath = os.path.join(folderPath, 'cache.json')
try:
    with open(cachePath, 'r') as f:
        cache = json.load(f)
except FileNotFoundError:
    with open(settingPath, 'w') as f:
        json.dump({}, f, ensure_ascii=False, indent=1)
        cache = {}

# Api Key
def changeValue(key, text):
    while True:
        inputText = input(text)
        settings[key] = inputText
            
        inputText = input("Confirm? (y/n): ")
        if inputText.lower() == "y" or inputText == "":
            break
        elif inputText.lower() != "n":
            print("Input a valid value")

if not settings['civitai_api_key']:
    changeValue('civitai_api_key', 'API Key: ')

#Input
if not settings['models_path']:
    while True:
        changeValue('models_path', 'Input Path: ')

        if os.path.exists(settings['models_path']):
            if os.path.isfile(settings['models_path']):
                settings['models_path'] = os.path.dirname(settings['models_path'])
            break
        else:
            print("Input a valid input path")

# Save Settings
with open(settingPath, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=1)

files = []
filetypes = ('*.safetensors', '*.ckpt')
for ext in filetypes:
    files.extend(glob.glob(os.path.join(settings['models_path'], '**', ext), recursive=True))

fileCount = len(files)
lenCount = len(str(fileCount))
print(f"\nfound: {fileCount} models\n")

data = {}
for i, file in enumerate(files):
    
    fileName = os.path.split(file)[1]
   
    if fileName in cache:
        sha256 = cache[fileName]['SHA256']
    else:
        print(f'{i+1:>{lenCount}d}/{fileCount} caching hash {fileName} -> ',end="")
        sha256 = calculate_sha256(file).upper()
        cache[fileName] = {}
        cache[fileName]['SHA256'] = sha256
        cache[fileName]['type'] = None
        print(f'{sha256}')
    
    data[sha256] = {}
    data[sha256]['path'] = file
    data[sha256]['fileName'] = fileName
    data[sha256]['data'] = {}

save_cache()

# split in chunk of 100
hashes = [value for value in data.keys()]
if len(hashes) != 0:
    print(f"\nRequesting: {len(hashes)} models from Civitai.com")

results = []
try:
    for i in range(0, len(hashes), 100):
        batch = hashes[i:i + 100]
        results.extend(req(f'/model-versions/by-hash', method='POST', data=batch))
except:
    print('Failed to fetch hash from Civitai')
    input('press "enter" to exit')
    exit()

print(f"Matched: {len(results)} models\n")

for result in results:
    found = False
    for file in result['files']:
        sha256 = file['hashes']['SHA256']
        if not sha256 in data:
            continue

        data[sha256]['data'] = result
        found = True
        break

    if not found:
        print(f"ERROR: {result['model']['name']}")

dataCount = len(data)
lenCount = len(str(dataCount))
for i, key in enumerate(data):
    model = data[key]

    counter = f'{i+1:>{lenCount}d}/{dataCount}'

    if model["data"] == {}:
        path = os.path.relpath(model['path'], start=settings['models_path'])
        print(f'{counter} Could not find Civitai Data for "{path}"')
        continue
    
    fileName = os.path.splitext(model['fileName'])[0]
    
    modelName = model['data']['model']['name']
    modelType = model['data']['model']['type']
    modelID = model['data']['modelId']

    versionName = model['data']['name']
    versionID = model['data']['id']
    
    newPath = os.path.join(folderPath, modelType, fileName)

    os.makedirs(os.path.join(folderPath, modelType), exist_ok=True)
    os.makedirs(newPath, exist_ok=True)
    cache[model['fileName']]['type'] = modelType

    url = f"https://civitai.com/models/{modelID}?modelVersionId={versionID}"

    # Lazy way to parse the HTML descriptions
    if model['data']['description']:

        description = model['data']['description']

        description = description.replace('<p>', '').replace('</p>', '\n')
        description = re.sub(r'<br[^>]*>', '\n', description)
        description = description.replace('<li>', '\t').replace('</li>', '')

        while True:
            m = re.search(r'<a.*?<\/a>', description)
            if not m:
                break

            matchText = m.group(0)
            matchURL = re.search(r'href="([^"]*)">', matchText).group(1)
            description = f'{description[:m.start(0)]}{matchURL}{description[m.end(0):]}'
        
        description = re.sub(r'<[^>]*>', '', description)

        description = f"\n\nVersion Description:\n{description}"
    else:
        description = ''

    if len(model['data']['trainedWords']) != 0:
        triggerWords = f"\n\nTrigger Words:\n{', '.join(model['data']['trainedWords'])}"
    else:
        triggerWords = ""

    # info txt file
    newImages = []
    if len(model['data']['images']) != 0:
        prompts = ''
        for image in model['data']['images']:
            
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
            examplePrompt = f'\n\nExample Prompts:{prompts}'
        else:
            examplePrompt = ''
    else:
        examplePrompt = ''

    with open(os.path.join(newPath, f'{fileName}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Model type:   {modelType}\nModel Name:   {modelName}\nVersion Name: {versionName}\n{url}{description}{triggerWords}{examplePrompt}')
    

    newImagesCount = len(newImages)

    # Download images preview files
    if newImagesCount != 0:
        print(f'\n{counter} Found {modelType} "{modelName}" version "{versionName}" on Civitai with {newImagesCount} preview images')

        with tqdm(total=newImagesCount, unit='Images') as bar:
            for image in newImages:
                response = requests.get(image['url'], stream=True, headers={'User-Agent': user_agent})

                image_data = BytesIO()

                for chunk in response.iter_content(chunk_size=download_chunk_size):
                    if chunk:
                        image_data.write(chunk)

                pil_image = Image.open(image_data)
                metadata = PngInfo()
                if image['pngInfo'] != '':
                    metadata.add_text('parameters', image['pngInfo'])
                    
                pil_image.save(image['path'], format='PNG', pnginfo=metadata, compress_level=4)

                bar.update(1)

# Remove old folder/cache
print()
for file, item in cache.items():
    if item['SHA256'] in data:
        continue
    if not item['type']:
        continue
    path = os.path.join(folderPath, item['type'], os.path.splitext(file)[0])
    if os.path.exists(path):
        print(f'"{file}" has been removed by user, deleting "{item["type"]}\{os.path.splitext(file)[0]}"')
        shutil.rmtree(path)

save_cache()
input('\nDone (press "enter" to exit)')
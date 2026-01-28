# VideoGPA

VideoGPA is a video generation quality assessment and optimization framework with DPO (Direct Preference Optimization) training capabilities.

# Quick Inference Scripts 🚀

This directory contains simplified command-line scripts for generating videos using **CogVideoX** models. These scripts are designed for quick testing and allow you to run inference directly from the terminal without preparing complex JSON configuration files.

Both scripts support loading **LoRA adapters** for customized generation.

## 📋 Requirements

Please make sure your Python version is between 3.10 and 3.12, inclusive of both 3.10 and 3.12.

```bash
pip install -r requirements.txt
```
## 🔘 Checkpoint Download 

### Automatic Download (Recommended)
#### Method 1: Using the Download Script
Run the provided Python script to automatically download all checkpoint files:
```bash
python download_checkpoints.py
```

The script will:
- ✅ Check if files already exist (skip re-downloading)
- 🚀 Download missing checkpoints with progress bars
- 📁 Organize files into the correct directory structure
- ⚡ Resume failed downloads automatically

### Expected Directory Structure
After successful download, your checkpoints folder should look like:
```
checkpoints/
├── VideoGPA-I2V-lora/
│   └── adapter_model.safetensors
└── VideoGPA-T2V-lora/
    └── adapter_model.safetensors
```

## 📝 Available Scripts

### 1. Text-to-Video Generation ([t2v_inference.py](generate/t2v_inference.py))

Generate videos from text prompts using CogVideoX-5B.

**Basic Usage:**
```bash
cd generate
python t2v_inference.py "A cat playing with a ball in a garden"
```

**Advanced Usage:**
```bash
python t2v_inference.py "A flying drone over a city skyline at sunset" \
    --output_dir ./my_videos \
    --lora_path ./checkpoints/my_lora_adapter \
    --gpu_id 0
```

**Arguments:**
- `prompt` (required): Text prompt for video generation
- `--output_dir`: Directory to save generated videos (default: `./outputs`)
- `--lora_path`: Path to LoRA adapter weights (optional)
- `--gpu_id`: GPU device ID (default: 0)

**Output:** Videos saved as `{prompt}_seed{seed}.mp4`

---

### 2. Image-to-Video Generation ([i2v_inference.py](generate/i2v_inference.py))

Generate videos from a static image with text guidance using CogVideoX-5B-I2V.

**Basic Usage:**
```bash
cd generate
python i2v_inference.py "The camera slowly zooms in" ./path/to/image.jpg
```

**Advanced Usage:**
```bash
python i2v_inference.py "A realistic continuation of the reference scene. Everything must remain completely static: no moving people, no shifting objects, and no dynamic elements. Only the camera is allowed to move. Render physically accurate multi-step camera motion.  Camera motion: roll gently to one side, then swing around the room, followed by push forward into the scene." ./image.png \
    --output_dir ./i2v_outputs \
    --lora_path ./checkpoints/i2v_lora \
    --gpu_id 1
```

**Arguments:**
- `prompt` (required): Text prompt describing motion/scene
- `image_path` (required): Path to input image file
- `--output_dir`: Directory to save generated videos (default: `./outputs`)
- `--lora_path`: Path to LoRA adapter weights (optional)
- `--gpu_id`: GPU device ID (default: 0)

**Output:** Videos saved as `{image_name}_seed{seed}.mp4`

---

## ⚙️ Configuration

Both scripts include configurable generation parameters:

```python
NUM_INFERENCE_STEPS = 50  # Number of diffusion steps
GUIDANCE_SCALE = 6.0      # Classifier-free guidance scale
SEED = 42              # Seed for generation
```


## 💾 GPU Memory Requirements

- **Minimum VRAM**: diffusers BF16 ~5GB for base models 
- Memory optimizations (VAE tiling/slicing) are automatically enabled

## 🎬 Visual Comparisons

<video src="https://github.com/user-attachments/assets/40bfebaf-365c-48f0-90dc-ee574228024a" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Pirate-themed amusement rides in a serene outdoor park...</summary>
  <br>
  <blockquote>
     The video features a series of pirate-themed amusement rides in an outdoor park setting, with each ride having unique names like 'Pirate Ship,' 'Pirate's Bay,' 'Pirate's Plunder,' 'Pirate's Cove,' 'Pirate's Revenge,' 'Pirate's Castle,' 'Pirate's Bay,' 'Pirate's Plunder,' 'Pirate's Castle,' and 'Pirate's Plunder.' The rides are adorned with vibrant colors, decorative elements, and safety signs, including a 'No Entry' sign and a 'Safety' sign. The surrounding area is lush with trees, and the atmosphere is serene, with no people present. The video captures the tranquil and still ambiance of the park.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/2a603254-6270-4ada-b330-5fcb87f7edbf" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Modern glass building in urban setting surrounded by vehicles...</summary>
  <br>
  <blockquote>
     The video features a modern glass building with a reflective facade, situated in an urban setting with a mix of older brick buildings. The building's entrance is marked by a sculpture and a sign, with a black SUV and other vehicles parked in front. As the video progresses, the building's surroundings are shown under a partly cloudy sky, with a red stop sign, blue trash bins, and a street lamp visible. The scene includes a black pickup truck, a silver sedan, and a red SUV parked in the foreground, with the building's entrance and a statue atop a cylindrical structure highlighted. The video concludes with a black SUV parked in front of the building, with a green street sign and a blue dumpster nearby.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/ded40c82-f320-47f6-86a8-bb4bfbb1df1e" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Modern interior with marble staircase and science-themed displays...</summary>
  <br>
  <blockquote>
     The video explores a modern building's interior, starting with a view of a staircase with dark marble walls and a colorful light installation. As the camera moves, it reveals various angles of the staircase, the surrounding walls, and the building's contemporary design, including a glass balustrade and a 'Science and Technology' sign. The focus shifts to a hallway with a 'Transforming Lives Through Chemistry' display, a white table, and a blue wall with a green sign. The video concludes with a view of a grand staircase leading to an upper level, with a 'EXIT' sign and a 'Parking Garage' sign visible.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/3c902166-97b0-4fe9-8330-001cdc4ab1e6" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Residential AC maintenance and utility boxes in suburban Ohio...</summary>
  <br>
  <blockquote>
   The video shows a series of residential scenes in Westerville, OH, focusing on the maintenance of air conditioning units and utility boxes. Initially, two gray air conditioners are seen on a concrete slab, connected to a white electrical box and a gas meter, with a well-kept lawn and a wooden fence in the background. As the video continues, the number of units increases to four, and the condition of the pipes and electrical boxes varies, indicating regular upkeep. A weathered white gas valve and a rusted red brick wall with a white electrical box are also featured, alongside a neatly trimmed bush and a parked car, suggesting a suburban setting.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/4c6dd7f2-bfb4-497d-b524-deb1453eb0c0" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Vending machines and kiosks in a quiet airport terminal...</summary>
  <br>
  <blockquote>
     The video features a series of vending machines in an airport terminal, including a colorful lottery machine, a FedEx self-service kiosk, and various other machines like a Pepsi vending machine and a UPS store. The machines are set against a backdrop of modern architecture with reflective floors, bright lighting, and signs indicating accessible restrooms and elevators. The area is quiet and well-lit, with no visible activity, suggesting a moment of stillness in a bustling transit space. The video captures the essence of a busy yet orderly airport environment.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/bef53ca6-88b2-4509-83f7-cc6a93c18cca" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Tranquil Chinese gazebo in a park with winding brick pathway...</summary>
  <br>
  <blockquote>
     A traditional Chinese-style wooden gazebo with a curved roof and ornate latticework is situated in a well-maintained park, surrounded by lush greenery and a winding brick pathway. The scene is tranquil, with a tennis court visible in the background, suggesting a recreational setting. As time passes, the gazebo remains a focal point, with the surrounding trees and the pathway leading to it. The lighting indicates it might be early morning or late afternoon. A modern trash bin with a red and black design appears, indicating an emphasis on cleanliness. The park is serene, with no people or animals present.
  </blockquote>
</details>


<video src="https://github.com/user-attachments/assets/7152d981-8af4-45ce-bd3e-1eb39ace2c9e" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Serene urban park with autumnal trees and distant high-rise...</summary>
  <br>
  <blockquote>
   The video features a tranquil park with a variety of trees, including a prominent magnolia and others with autumnal leaves, set against an overcast sky. The park is devoid of people and wildlife, creating a serene atmosphere. As time passes, the scene includes a young magnolia with reddish-brown leaves and a backdrop of leafless trees, suggesting a seasonal transition. The presence of a watermark 'bilibili' indicates the footage may be from a Chinese video platform. The video concludes with a view of a grassy field with trees showing signs of autumn, and a distant high-rise building, hinting at an urban park setting.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/79242745-b24e-400e-826e-cb1f2f933342" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Vibrant video game store interior with collectibles and gaming posters...</summary>
  <br>
  <blockquote>
     The video takes us through a vibrant video game store, starting with a view of the store's interior, showcasing DVDs, video games, and collectibles. As we move through the store, we see a variety of items including baseball caps, plush toys, and action figures, with clearance signs indicating sales. The store features a range of posters, including 'TRENDPOSTERS' and 'SUPERSTAR POSTERS', and a section dedicated to gaming accessories. The shelves are well-organized, displaying DVDs, Blu-rays, and video games, with a focus on the entertainment industry. The store's atmosphere is inviting, with a warm color palette and a sense of excitement for customers.
  </blockquote>
</details>


<video src="https://github.com/user-attachments/assets/0cd77642-065b-4998-9dab-8c7e6f215282" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Outdoor dining area with black canopy and red cushioned chairs...</summary>
  <br>
  <blockquote>
      The video features an outdoor dining area with a black metal truss canopy and greenery, set against a red wall with large windows reflecting the interior. Initially, the area is empty, with a red pickup truck parked outside. As time passes, the scene includes a red bench, a black table with chairs, and a high-top table with red cushioned chairs. The area is well-lit by pendant lights and natural light, with a 'NO ENTRY' sign and a 'CLOSED' sign indicating the establishment's status. The setting is tranquil, with a red pickup truck and a black car parked nearby, and a 'PARKING' sign suggests restrictions.
</details>

<video src="https://github.com/user-attachments/assets/3d4c87bf-edf9-4f86-90db-dabfac6efcb3" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>Well-organized warehouse with stocked shelves and industrial lighting...</summary>
  <br>
  <blockquote>
      The video shows a warehouse with metal shelves initially empty, with a yellow 'Y' sign and a ladder suggesting recent activity. As time passes, the shelves are seen stocked with car batteries, and a hand truck appears, indicating ongoing work. The warehouse is well-lit, with a red wall and a pallet jack, but no workers are visible. The shelves are neatly organized, and a pallet jack is present, suggesting movement of goods. A black curtain partially conceals the shelves, and a red curtain adds a dramatic flair. The scene is quiet, with a few batteries and a pallet jack, and a red curtain partially veils the shelves, with a few batteries and a pallet jack, and a sign reading 'AUTO PARTS' in the background.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/cb02abbe-4bc3-4040-b131-2ec4bad92ecf
" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Monument to the People's Heroes in Beijing under sunlight...</summary>
  <br>
  <blockquote>
     The video features the Monument to the People's Heroes in Beijing, China, under a clear blue sky. The monument, with its grey stone base and golden Chinese characters, is surrounded by a white railing and lush greenery. As the sun casts a soft glow, the scene remains tranquil and devoid of people, emphasizing the monument's solemnity and historical significance. The inscription on the monument's base changes slightly, indicating different eras or aspects of the monument's history. The serene atmosphere is maintained throughout, with the monument's grandeur highlighted by the sunlight.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/2ce897d7-cac5-416c-9fe7-05b7ec83fe47" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Modern furniture store interior showcasing stylish sofas and decorative pieces...</summary>
  <br>
  <blockquote>
     The video takes viewers through a furniture store, showcasing a variety of modern and stylish furniture pieces. Initially, a white sectional sofa with plush cushions is highlighted, surrounded by a dark wood coffee table and a shelving unit with decorative items. As the video continues, different sections of the store are featured, including a cream-colored sofa with a chaise lounge, a round wooden coffee table, and a shelving unit with vases and books. The store's ambiance is warm and inviting, with a focus on contemporary furniture and decorative elements like potted plants and abstract art. The video concludes with a display of a white ceramic pot with green succulents on a wooden stool, surrounded by other furniture pieces.
  </blockquote>
</details>


<video src="https://github.com/user-attachments/assets/32e1183b-33ca-482a-b2a4-9d9a75127fcf" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Bronze statues of guardians and children in a tranquil park...</summary>
  <br>
  <blockquote>
      The video features a series of bronze statues in a park, each depicting an adult and a child in various intimate moments, suggesting a theme of guardianship and learning. The statues are set against a backdrop of a wooden bridge, lush greenery, and a clear sky, with a pathway leading to the bridge. The scenes are tranquil, with no people present, and the soft lighting indicates it might be early morning or late afternoon. The presence of a logo in some frames hints at an association with a media outlet or organization.
  </blockquote>
</details>


<video src="https://github.com/user-attachments/assets/fc8c5b53-7697-4645-93f4-101eab465203" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>   Modern urban sculpture in a quiet cityscape with maintenance equipment...</summary>
  <br>
  <blockquote>
   The video features a serene urban setting with a red and black abstract sculpture on a concrete base, surrounded by modern buildings and a clear blue sky. As time passes, the scene includes a cherry picker, a 'Parkway' sign, and a 'Downtown' sign, suggesting maintenance or installation activities. The sculpture's dynamic design contrasts with the tranquil environment. A cherry picker lifts a street sign, and a 'Parkway' sign is visible. The cityscape is quiet, with no people, and a 'Chase' building is seen in the background. The video concludes with a 'PARK' sign and a 'Chase' building, with a cherry picker indicating maintenance work.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/9c75949c-1aeb-4239-8184-afb5ee09dfd4" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>   Modern furniture store showcasing beds, living areas, and whimsical decor...</summary>
  <br>
  <blockquote>
    The video takes place in a modern furniture store, showcasing a variety of beds with different headboards and mattresses, some wrapped in plastic. The store features a dining set, a kitchenette, and a living area, all under warm lighting. Decorative elements like abstract art, a pink flamingo sculpture, and a plush ostrich toy add whimsy. The store displays beds with blue and white branding, including 'OYO' and 'QUBO', and offers promotional materials. The ambiance is contemporary, with a color scheme of neutral tones and warm lighting, creating an inviting atmosphere for customers.
  </blockquote>
</details>


<video src="https://github.com/user-attachments/assets/8d8e83f6-70c2-4aeb-9787-207e3e2a418c" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b> Indoor corridor with educational posters and blue floor stickers...</summary>
  <br>
  <blockquote>
   The video takes place in a well-lit indoor corridor with a polished floor and high ceilings, adorned with potted plants and educational posters. A red banner with Chinese characters is visible, along with a bulletin board displaying information and photographs. The corridor is empty, with blue social distancing stickers on the floor and a sign indicating the entrance to a building. As the video continues, the corridor is shown with a large window, a door with a red sign, and a staircase. The walls feature colorful murals and a 'Welcome' sign, with a few individuals seen in the distance. The setting suggests a school or community space with a focus on safety and orderliness.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/1bbf5217-5feb-4b02-9264-8e502f37955d" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>   Tranquil lakeside gazebo with lotus flowers and city skyline...</summary>
  <br>
  <blockquote>
    The video features a tranquil lakeside scene with a wooden boardwalk leading to a gazebo, surrounded by lotus flowers and aquatic plants. The calm waters reflect the soft glow of the sun, either rising or setting, and the hazy skyline of a city with modern skyscrapers. The scene is serene, with no people or wildlife, and the lighting suggests it's either early morning or late afternoon. As time passes, the sky transitions from pale blue to warm orange, and the city's silhouette becomes more pronounced against the natural backdrop. The video captures the peaceful coexistence of nature and urban life.
  </blockquote>
</details>


<video src="https://github.com/user-attachments/assets/eec48773-bb0f-47a6-aab0-03989cf2fabb" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>   Department store Halloween section with festive orange and black decor...</summary>
  <br>
  <blockquote>
    The video takes us through a department store's Halloween section, showcasing a variety of festive decorations. Initially, the store is empty, but as we move through the aisles, we see Halloween-themed items like candles, ornaments, and kitchenware. The decorations are arranged on wooden tables and shelves, with a color scheme of black, orange, and white. Some items feature playful phrases like 'BOO', 'BOOZE', and 'HOME SWEET HALLOWEEN'. The store is well-lit, with a warm ambiance, and the decorations include ghost figures, pumpkins, and bats, creating a cozy, celebratory atmosphere.
  </blockquote>
</details>


<video src="https://github.com/user-attachments/assets/f399ec4c-6b8c-4e21-b6ba-b3562038749c" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>   Industrial workshop featuring bus maintenance and hydraulic lifts...</summary>
  <br>
  <blockquote>
   The video takes place in a well-lit industrial workshop with a focus on bus maintenance. Initially, a blue and white hydraulic lift is central, with a red banner and a bus under maintenance. The scene shifts to show the lift with a control panel, a red banner with Chinese characters, and a bus with a damaged front end. Various equipment like a manual pallet jack, a red fire extinguisher, and a wooden bench appear, indicating ongoing work. A technician is seen working on a bus, with tools and parts scattered around. The workshop is equipped with multiple lifts, a forklift, and a bus undergoing repair, with a red banner and a sign with Chinese characters visible. The video concludes with a still scene of the workshop, highlighting the bus and the mechanical equipment.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/2472430f-f2bb-4608-9b1f-03c141c5dc2b" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>   Traditional Chinese tea room with elegant furniture and ceramic decor...</summary>
  <br>
  <blockquote>
    The video features a serene and culturally rich setting, showcasing a traditional Chinese tea ceremony room with a polished wooden table, elegant chairs, and a shelving unit with ceramic vases and books. As time passes, the room is shown with various angles, highlighting the tea set, the bamboo runner, and the soft lighting that creates a tranquil atmosphere. Decorative elements like a potted plant, a small statue, and framed pictures add to the ambiance. The room's design, with its minimalist aesthetic and soft lighting, suggests a space for contemplation and cultural appreciation.
  </blockquote>
</details>


<video src="https://github.com/user-attachments/assets/ccc1f9ed-2971-4d73-b005-c14e5ec07e0d" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>   Traditional Chinese museum featuring golden artifacts and historical dioramas...</summary>
  <br>
  <blockquote>
   The video explores a traditional Chinese museum, starting with a display case of golden figurines and moving through various historical exhibits, including a lion statue, a model of an ancient Chinese city, and a human skeleton with tools. The museum features a high ceiling with wooden beams, stone columns, and bamboo wall panels, with soft lighting enhancing the cultural ambiance. Textual information in Chinese is visible, suggesting an educational purpose. The scenes transition from a serene, well-preserved interior to a more somber, reflective atmosphere with a display case of golden artifacts and a model of a historical battle scene, all under a warm glow.
  </blockquote>
</details>

<video src="https://github.com/user-attachments/assets/96141bb4-5874-4101-8596-e5ce2a9d0b77" width="100%" controls preload="metadata"></video>
<details>
  <summary><b>Prompt:</b>   Well-organized warehouse with stocked shelves and industrial lighting...</summary>
  <br>
  <blockquote>
    The video takes us through a well-lit warehouse filled with shelves stocked with various items. Initially, we see a red wall with metal shelves holding boxes, bags, and a ladder. As we move forward, the shelves display an assortment of goods including kitchen appliances, cleaning supplies, and automotive parts. The items are neatly organized, with some boxes labeled for specific products. A hand truck is visible, suggesting recent or upcoming activity. The warehouse is brightly lit, with fluorescent lights enhancing the visibility of the goods, which range from new to used, and are methodically arranged for easy access and identification.
  </blockquote>
</details>

## 🚀 Features

- **Video Quality Assessment**: Comprehensive metrics for evaluating video generation quality
- **DPO Training**: Direct Preference Optimization for video generation models
- **Multi-Model Support**: Compatible with CogVideoX and other video generation models
- **Flexible Pipeline**: Easy-to-use inference and training pipelines

## 📁 Code Structure

```
VideoGPA/
├── data_prep/      # Data preparation scripts
├── train_dpo/      # DPO training scripts
├── pipelines/      # Inference pipelines
├── metrics/        # Quality assessment metrics
├── vggt/           # Video generation model architecture
└── utils/          # Utility functions
```


## 🔧 DPO Training (Direct Preference Optimization) 

VideoGPA leverages DPO to optimize video generation quality through preference learning. The training pipeline consists of 3 steps after you have your generated videos. Revise the configs as you need:

#### Step 1: Score Your Generate Videos
```bash
python train_dpo/video_scorer.py
```

#### Step 2: Encode Videos to Latent Space
```bash
# For Text-to-Video models
python train_dpo/CogVideoX-T2V-5B_lora/02_encode.py

# For Image-to-Video models
python train_dpo/CogVideoX-I2V-5B_lora/02_encode.py
```

#### Step 3: Run DPO Training
```bash
# Text-to-Video DPO training
python train_dpo/CogVideoX-T2V-5B_lora/03_train.py

# Image-to-Video DPO training
python train_dpo/CogVideoX-I2V-5B_lora/03_train.py
```

**Key Features:**
- 🎯 Preference-based learning using winner/loser pairs
- 🔧 Parameter-efficient fine-tuning with LoRA
- 📊 Multiple quality metrics support
- ⚡ Distributed training with PyTorch Lightning
- 💾 Automatic gradient checkpointing and memory optimization

**Data Format:** Training requires JSON metadata containing preference pairs - multiple videos generated from the same prompt with quality scores. See [dataset.py](train_dpo/dataset.py) for details.


## 🙏 Acknowledgements

Built on top of CogVideoX and other state-of-the-art video generation models.

## 🌟 Citation
If you find our work helpful, please leave us a star and cite our paper.


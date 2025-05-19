#  User-Guided Image Colorization
This project is an extension of the SIGGRAPH2017 image colorization model by Zhang et al., adding a user interface that allows scribble-based color hints to guide the colorization process.

⸻

🚀 Features
	•	Interactive OpenCV GUI for scribble-based color hints
	•	Modern PyQt5 color picker
	•	Support for hinted ab input and hint masks
	•	Integration with pretrained SIGGRAPH2017 model
	•	Fully automated pipeline: draw → colorize → save

# Install dependencies
pip install torch torchvision opencv-python numpy matplotlib PyQt5

# Folder Structure

project_root/
│
├── imgs/                        # Input grayscale images from the original work
├── imgs_out/				     # Outputs from the initial demo
├── test_img						 # Inputs for the extension
├── saved_outputs/               # Output colorized results from the extension work
├── colorizers/                  # Model definition (includes siggraph17.py)
│
├── gui_draw.py                  # Scribble GUI + color picker
├── initial_demo.py              # Automatic colorization(without hint)
├── ab_hint_map.npy              # (auto-generated) ab hint map
├── hint_mask.npy                # (auto-generated) binary mask
├── demo_ext.py                  # Our extention of the model

# How to use
python gui_draw.py --img_path imgs/your_image.jpg

Press p to open color picker
	•	Click to draw on the image
	•	Press s to save hints and automatically trigger colorization
	•	Press q to quit

The GUI saves:
	•	ab_hint_map.npy
	•	hint_mask.npy
	•	Automatically runs demo_ext.py afterward

# Model-Based Colorization
python initial_demo.py --img_path imgs/your_image.jpg --save_prefix saved_outputs/your_output

This loads the grayscale image, hints, and runs the SIGGRAPH2017 model
	• Output saved as:
        saved_outputs/your_output_siggraph17.png

 Notes
	•	Make sure your ab_hint_map.npy and hint_mask.npy match the same image used in --img_path
	•	If using a GPU:
            python demo_release.py --use_gpu

Credits

Based on:
	•	Zhang et al. (2017): “Real-Time User-Guided Image Colorization with Learned Deep Priors”
	•	Extended with custom scribble UI, PyQt5 integration, and full inference automation
# dream-cli
 
CLI for a deepdream-implementation. Place the image(s) you want to render into the "Images" folder and run dreamcli.py

# Settings

Octaves: How many times the source image gets scaled down by octave_scale. 

Iterations: How many times the gradient is calculated and added to the image per octave.

Iteration descent: Each octave the Iteration descent is subtracted from the total iteration count. 

Background: Writes the accumulated gradient to a monocolored (background color) image

# Renderer

Each Renderer calculates the gradient for a specific layer[channels] with the corrosponding renderer-settings. Renderers can be mixed, the order of Renderer does not matter.

Layer & channels: 204 layers available, to render only the first channel of a layer, set channels to 0 and 1.

Squared: Wether the tensor is squared or not

Step size: The gradient gets multiplicated by the step_size each iteration. Higher values mean faster, but less accurate results. A negative step size results in a subtracted gradient. 

Cropped & Boundraries: Renders only a cropped area of the source image [[0,1],[0,1]] renders the whole image.

RGB-filter: Pixel values of the source image that are not in range of the RGB-filter get deleted from the gradient. Warning: Very slow.

Rotation: Most layers/channels are directional, meaning some patterns always point to a specific direction. Rotating the source image before calculating the gradient changes the direction.

Tile size: The inception model is trained for small (~300x300 pixel) images. The source image is split into multiple tiles before calculating the gradient. Increasing the tile size increases memory usage and decreases accuracy. 

Color multipler: The gradient gets multiplied with the color multiplier before added to the image. A color multiplier with values [2,1,0] boosts the red amount of the gradient and completly deletes any blue amount in the gradient. Changes the pattern. Acccepts negative values


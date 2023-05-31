from PIL import Image

# Load the image
image = Image.open("hello.png").convert("L")
width, height = image.size
pixels = list(image.getdata())

histogram = [0] * 256
for pixel in pixels:
    histogram[pixel] += 1

cdf = [0] * 256
cdf[0] = histogram[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + histogram[i]

min_cdf = min(cdf)
max_cdf = max(cdf)

output_pixels = []
for pixel in pixels:
    new_pixel = int((cdf[pixel] - min_cdf) / (width * height - min_cdf) * 255)
    output_pixels.append(new_pixel)

output_image = Image.new("L", (width, height))
output_image.putdata(output_pixels)
output_image.save("kitty.png")

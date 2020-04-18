% CLAHE (Contrast Limited Adaptive Histogram Equalization) filtering
function enhance_image = clahe(image)
     enhance_image = adapthisteq(image);
end

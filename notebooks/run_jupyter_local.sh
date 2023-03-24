# python -m ipykernel install --user --name wct2 --display-name "Python wct2"
cd ../../ && LOG=INFO CUDA_VISIBLE_DEVICES=$1 jupyter-notebook --ip="*" --no-browser --allow-root --port $2

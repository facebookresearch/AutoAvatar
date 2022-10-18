data_dir=$1
start_i=$2

ffmpeg -start_number $start_i -i $data_dir/gt_imgs/%06d.jpg -c:v libx264 -vf "fps=25,format=yuv420p" $data_dir/gt.mp4
ffmpeg -start_number $start_i -i $data_dir/pred_imgs/%06d.jpg -c:v libx264 -vf "fps=25,format=yuv420p" $data_dir/pred.mp4
ffmpeg -start_number $start_i -i $data_dir/errs_imgs/%06d.jpg -c:v libx264 -vf "fps=25,format=yuv420p" $data_dir/errs.mp4
# ffmpeg -start_number $start_i -i $data_dir/pred_cano_imgs/%06d.jpg -c:v libx264 -vf "fps=25,format=yuv420p" $data_dir/pred_cano.mp4

# ffmpeg -i $data_dir/gt.mp4 -i $data_dir/pred.mp4 -i $data_dir/errs.mp4 -i $data_dir/pred_cano.mp4 -filter_complex hstack=inputs=4 $data_dir/concat.mp4
ffmpeg -i $data_dir/gt.mp4 -i $data_dir/pred.mp4 -i $data_dir/errs.mp4 -filter_complex hstack=inputs=3 $data_dir/concat.mp4

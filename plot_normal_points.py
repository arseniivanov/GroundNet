from data_loader import visualize_frame_annotation


annotation_file = "book_annotations_train\\book_batch-1_4_annotation.pbdata"
video_filename = "book_annotations_train\\book_batch-1_4_video.MOV"

visualize_frame_annotation(annotation_file, video_filename, resize=True)
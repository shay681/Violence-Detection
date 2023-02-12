from keras.layers import Dense, Input, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from hierarchical_attention_network import HanAttentionLayer
from keras.optimizers import Adam


def compile_model(n_videos, max_poses, max_frames, lr, print_summary=False):
    print(n_videos)
    sentence_input = Input(shape=(max_poses,51))
    linear_rep = Dense(64, activation='relu')(sentence_input)
    pose_annotation = Bidirectional(GRU(64, return_sequences=True))(linear_rep)
    print(f'pose_annotation: {pose_annotation.shape}')
    attn_pose = HanAttentionLayer(64)(pose_annotation)
    print(f'attn_pose: {attn_pose.shape}')

    frameEncoder = Model(sentence_input, attn_pose)

    video_input = Input(shape=(max_frames, max_poses, 51))
    print(f'video_input: {video_input.shape}')

    video_encoder = TimeDistributed(frameEncoder)(video_input)
    print(f'video_encoder: {video_encoder.shape}')
    linear_rep_frame = Dense(64, activation='relu')(video_encoder)
    frame_annotation = Bidirectional(GRU(64, return_sequences=True))(linear_rep_frame)
    attn_frame = HanAttentionLayer(64)(frame_annotation)
    preds = Dense(2, activation='softmax')(attn_frame)
    model = Model(video_input, preds)

    opt = Adam(learning_rate=lr)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    if print_summary:
        model.summary()

    return model

def get_model_parts(dense_dim: int):
    MAX_FRAMES = 151
    MAX_POSES = 20
    sentence_input = Input(shape=(MAX_POSES,51))
    linear_rep = Dense(dense_dim, activation='relu')(sentence_input) # Dense 64 dim
    pose_annotation = Bidirectional(GRU(64, return_sequences=True))(linear_rep) 
    print(f'pose_annotation: {pose_annotation.shape}')
    attn_pose = HanAttentionLayer(64)(pose_annotation)
    print(f'attn_pose: {attn_pose.shape}')

    frameEncoder = Model(sentence_input, attn_pose)

    video_input = Input(shape=(MAX_FRAMES, MAX_POSES, 51))
    print(f'video_input: {video_input.shape}')

    video_encoder = TimeDistributed(frameEncoder)(video_input)
    print(f'video_encoder: {video_encoder.shape}')
    linear_rep_frame = Dense(dense_dim, activation='relu')(video_encoder) # Dense 64 dim
    frame_annotation = Bidirectional(GRU(64, return_sequences=True))(linear_rep_frame)
    attn_frame = HanAttentionLayer(100)(frame_annotation)
    preds = Dense(2, activation='softmax')(attn_frame)
    return video_input, preds

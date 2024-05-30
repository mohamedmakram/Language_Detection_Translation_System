# Language_Detection_Translation_System


you can download the translation model from here:
https://drive.google.com/file/d/1-5kHQ_VottvfHSF6sKymvcDCq6ixo8vD/view?usp=sharing

to load this model ----------------> don't include the file name within the directory 
model = AutoModelForSeq2SeqLM.from_pretrained('\model path', use_safetensors=True)


with regard to language detection model, you can download the model from here:

https://drive.google.com/file/d/1VjohNW82kyKoKKVSbpcVLlrgJ7raz7rm/view?usp=sharing

to load the model
state_dict = torch.load('lang_detect.pth', map_location=torch.device('cpu'))


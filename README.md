## **MH-1M Dataset**

Introduction to MH-1M: A Comprehensive Dataset for Android Malware Detection

### **Introduction**

The rapid and widespread increase of Android malware presents substantial obstacles to cybersecurity research. In order to revolutionize the field of malware research, we present the MH-1M dataset, which is a thorough compilation of **1,340,515 APK** samples. This dataset encompasses a wide range of diverse attributes and metadata, offering a comprehensive perspective. The utilization of the VirusTotal API guarantees precise assessment of threats by amalgamating various detection techniques. Our research indicates that MH-1M is a highly current dataset that provides valuable insights into the changing nature of malware.  
  
MH-1M consists of 23,247 features that cover a wide range of application behavior, from intents::accept to apicalls::landroid/window/splashscreenview.remove. The features are categorized into four primary classifications:  


|Feature Types| Values  |
|--|--|
|APICalls  |22,394  |
|Intents  |407    |
|OPCodes  |232  |
|Permissions  |214  |

The dataset is stored efficiently, utilizing a memory capacity of **29.0 GB**, which showcases its substantial yet controllable magnitude. The dataset consists of **1,221,421 benign** applications and **119,094 malware** applications, ensuring a balanced representation for accurate malware detection and analysis.  

The MH-1M repository also offers a wide variety of metadata from APKs, providing useful data into the development of malicious software over a period of more than ten years. The Android features include a wide variety of metadata, which includes SHA256 hashes, file names, package names, compilation APIs, and various other details. This GitHub repository contains over 400GB of valuable data, making it the largest and most comprehensive dataset available for advancing research and development in Android malware detection.

Information about raw files:
1. Features (305G)
2. Labels (97G)
3. Metadata (5.3G)

Information about compressed files:
1. Features (46G)
2. Labels (12G)
3. Metadata (137M)

Link to drive containing Feature, Labels and Metadata: https://drive.google.com/drive/folders/1XUWOvG3CkiK7KbYAOofkxl1ST4-y_hhJ

  
The MH-1M's wide range of features and comprehensive coverage make it an essential tool for researchers seeking to comprehend and counter the constantly evolving Android malware threat landscape. With its detailed metadata and comprehensive feature representation, MH-1M is a crucial resource for advancing malware research and developing strong cybersecurity solutions.

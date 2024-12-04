# SMARTNESS (dataSet MalwAre fRom The wiNdows opErating SyStem)
Dataset of OpCodes from Real and Active Malware on the Windows Operating System

Initially, a broad collection of 20,371 malware samples was gathered from the VirusShare repository (https://virusshare.com/). As shown below, these samples were categorized into seven distinct malware families, aiming to cover a wide spectrum of malicious behaviors. The compatibility of the artifacts with the executable standard of the Microsoft Windows platform was verified using the PEFile library (https://github.com/erocarrera/pefile), a technical choice that ensures the relevance of the samples within the scope of this research.

Class   Family                               Qty     Download Link
1	      Backdoor:Win32/Bifrose	             1079    https://drive.google.com/file/d/1rDdboN6I8ATv3myT8NkISDu8bj3Qojcb/view?usp=sharing

2	      Trojan:Win32/Vundo	                 5644    https://drive.google.com/file/d/1zXuR8u1soYImTat-OchJ1fVjCdaFwVcB/view?usp=sharing

3	      BrowserModifier:Win32/Zwangi	        468    https://drive.google.com/file/d/1BCSgD9ulo77oIoadGk-7j7WUi6uNdfJj/view?usp=sharing

4	      Trojan:Win32/Koutodoor	             3937    https://drive.google.com/file/d/16chdG6CX8vV65vvDynjRPikEzbqjC6GG/view?usp=sharing

5	      Backdoor:Win32/Rbot	                  771    https://drive.google.com/file/d/1j-FTJi7Qk68wE8WDUxL0Epsrd2acDmh6/view?usp=sharing

6	      Backdoor:Win32/Hupigon	             1174    https://drive.google.com/file/d/1V4JEttIjKxugTcpxe3k0va2kYtRbuyJK/view?usp=sharing

7	      Trojan:Win32/Startpage	              646    https://drive.google.com/file/d/1X1NwYIp8ZKnLMxMwHqYz0nBijB8PAgyu/view?usp=sharing

             Total	                        13719

A Python program was developed to automate the malware disassembly process for the entire malware dataset. This program calculated the exact location of the first opcode of each malware by using the ”AddressOfEntryPoint” header field of the PE (Portable Executable) file. This step is crucial for the effective extraction of data from active malware, differentiating it from approaches that use inactive or inoperative samples.

The effectiveness of the script was corroborated through comparative analyses with the outputs produced by IDA Pro (https://hex-rays.com/ida-pro/), a market-leading disassembler, ensuring the reliability of the extracted data.

After disassembly, the focus shifts to extracting Opcode sequences from the malicious artifacts. This work exclusively concentrates on the Opcode sequences, such as "mov", "push", "call", "or", which are extracted and stored for subsequent analysis.

The concatenated files by class (referred to as summaries) are available at the link below:

https://drive.google.com/drive/folders/1SM_qG3IBSSM9ZSSaURFhB9jTGapf3K1g?usp=drive_link



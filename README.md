# SMARTNESS
Dataset de OpCodes de malwares reais e ativos do sistema operacional Windows
Inicialmente, optou-se por uma coleta ampla de 20.371 malwares, obtidos através do repositório VirusShare (https://virusshare.com/). Conforme pode ser visto abaixo, essas amostras foram categorizadas em sete famílias distintas de malwares, visando cobrir um espectro amplo de comportamentos maliciosos. A compatibilidade dos artefatos com o padrão executável da plataforma Microsoft Windows foi verificada utilizando a biblioteca PEFile (https://github.com/erocarrera/pefile), uma escolha técnica que assegura a pertinência das amostras dentro do escopo desta pesquisa.

Classe	      Família	                      Qt. (b)  Link para download

1	      Backdoor:Win32/Bifrose	             1079    https://drive.google.com/file/d/1rDdboN6I8ATv3myT8NkISDu8bj3Qojcb/view?usp=sharing

2	      Trojan:Win32/Vundo	                 5644    https://drive.google.com/file/d/1zXuR8u1soYImTat-OchJ1fVjCdaFwVcB/view?usp=sharing

3	      BrowserModifier:Win32/Zwangi	        468    https://drive.google.com/file/d/1BCSgD9ulo77oIoadGk-7j7WUi6uNdfJj/view?usp=sharing

4	      Trojan:Win32/Koutodoor	             3937    https://drive.google.com/file/d/16chdG6CX8vV65vvDynjRPikEzbqjC6GG/view?usp=sharing

5	      Backdoor:Win32/Rbot	                  771    https://drive.google.com/file/d/1j-FTJi7Qk68wE8WDUxL0Epsrd2acDmh6/view?usp=sharing

6	      Backdoor:Win32/Hupigon	             1174    https://drive.google.com/file/d/1V4JEttIjKxugTcpxe3k0va2kYtRbuyJK/view?usp=sharing

7	      Trojan:Win32/Startpage	              646    https://drive.google.com/file/d/1X1NwYIp8ZKnLMxMwHqYz0nBijB8PAgyu/view?usp=sharing

             Total	                        13719

Para a desmontagem dos malwares, foi desenvolvido um script Python, visando a automação desse processo para a nossa base de malwares. Este passo é crucial para a extração eficaz dos dados de malwares ativos, diferenciando-se de abordagens que utilizam amostras inativas ou inoperantes. A eficácia do script foi corroborada através de análises comparativas com as saídas produzidas pelo IDA Pro (https://hex-rays.com/ida-pro/), um desmontador líder de mercado, garantindo a fidedignidade dos dados extraídos.

Após a desmontagem, o foco se volta para a extração das sequências de Opcodes dos artefatos maliciosos. Neste trabalho, concentra-se exclusivamente na sequência de Opcodes, como `"mov", "push", "call", "or", extraída e armazenada para análises subsequentes.





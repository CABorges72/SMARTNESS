# SMARTNESS
Dataset de OpCodes de malwares reais e ativos do sistema operacional Windows
Inicialmente, optou-se por uma coleta ampla de 20.371 malwares, obtidos através do repositório VirusShare (https://virusshare.com/). Conforme pode ser visto abaixo, essas amostras foram categorizadas em sete famílias distintas de malwares, visando cobrir um espectro amplo de comportamentos maliciosos. A compatibilidade dos artefatos com o padrão executável da plataforma Microsoft Windows foi verificada utilizando a biblioteca PEFile (https://github.com/erocarrera/pefile), uma escolha técnica que assegura a pertinência das amostras dentro do escopo desta pesquisa.

Classe	      Família	                            Qt. (a)  Qt. (b)

1	      Backdoor:Win32/Bifrose	         2291     1079

2	      Trojan:Win32/Vundo	                6794	    5644

3	      BrowserModifier:Win32/Zwangi	          920	     468

4	      Trojan:Win32/Koutodoor	         5605	    3937

5	      Backdoor:Win32/Rbot	                1170	     771

6	      Backdoor:Win32/Hupigon	         1943	    1174

7	      Trojan:Win32/Startpage	         1648	     646

             Total	                             20371	   13719

Para a desmontagem dos malwares, foi desenvolvido um script Python, visando a automação desse processo para a nossa base de malwares. Este passo é crucial para a extração eficaz dos dados de malwares ativos, diferenciando-se de abordagens que utilizam amostras inativas ou inoperantes. A eficácia do script foi corroborada através de análises comparativas com as saídas produzidas pelo IDA Pro (https://hex-rays.com/ida-pro/), um desmontador líder de mercado, garantindo a fidedignidade dos dados extraídos.

Após a desmontagem, o foco se volta para a extração das sequências de Opcodes dos artefatos maliciosos. Neste trabalho, concentra-se exclusivamente na sequência de Opcodes, como `"mov", "push", "call", "or", extraída e armazenada para análises subsequentes.





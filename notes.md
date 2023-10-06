# Notes


### Forslag til første modell
- Lage validation-set av tilfeldig utvalg av treningsdatasettet basert på estimert vær.
- Samle resten av treningsdatasettet til ett datasett, der man legger til en feature som sier om raden har estimert vær eller observert vær
- manglende dataverdier må fylles på en eller annen måte
- trene modell uten å droppe noen features
- teste modell på validation set

### Fjerning av features
- Fjerne features som enten har veldig høy korrelasjon med andre features
- Fjerne features som har veldig lite å si for outcomet
- Fjerne features som er veldig dårlig estimert
- Fjerne features som kan bli utledet av andre features (høy korrelasjon med et utvalg andre features)

### Hvordan ta høyde for forskjell mellom estimert vær og observert vær
- Legge til boolean feature for om en rad er estimert eller observert vær
- Tenke på hvordan validation set skal lages

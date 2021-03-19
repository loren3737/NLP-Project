import data_plotter

kOutputDirectory = "D:\\Coding\\UW_PMP_Code\\NLP\\Project\\roc"

# RNN
rnn_pos = [1.0, 0.2234910277324633, 0.20880913539967375, 0.199836867862969, 0.19412724306688417, 0.19086460032626426, 0.18515497553017946, 0.18189233278955955, 0.17944535073409462, 0.17781402936378465, 0.17536704730831973, 0.17536704730831973, 0.1729200652528548, 0.16802610114192496, 0.16802610114192496, 0.16639477977161501, 0.16476345840130505, 0.1639477977161501, 0.16231647634584012, 0.16068515497553018, 0.1598694942903752, 0.1598694942903752, 0.15905383360522024, 0.15905383360522024, 0.15742251223491027, 0.1566068515497553, 0.15579119086460033, 0.15579119086460033, 0.15497553017944535, 0.15415986949429036, 0.15415986949429036, 0.1533442088091354, 0.15252854812398042, 0.15252854812398042, 0.15252854812398042, 0.15252854812398042, 0.15089722675367048, 0.15089722675367048, 0.14926590538336051, 0.14763458401305057, 0.1468189233278956, 0.1468189233278956, 0.1468189233278956, 0.14600326264274063, 0.14600326264274063, 0.14518760195758565, 0.14518760195758565, 0.14518760195758565, 0.14437194127243066, 0.14192495921696574, 0.14110929853181076, 0.14110929853181076, 0.13947797716150082, 0.13947797716150082, 0.13866231647634583, 0.13784665579119088, 0.1370309951060359, 0.1362153344208809, 0.1362153344208809, 0.1362153344208809, 0.13539967373572595, 0.13539967373572595, 0.13376835236541598, 0.132952691680261, 0.132952691680261, 0.132952691680261, 0.13050570962479607, 0.12969004893964112, 0.12969004893964112, 0.12969004893964112, 0.12887438825448613, 0.1264274061990212, 0.1264274061990212, 0.1264274061990212, 0.12561174551386622, 0.12561174551386622, 0.12479608482871125, 0.12398042414355628, 0.1231647634584013, 0.1231647634584013, 0.12234910277324633, 0.12153344208809136, 0.12153344208809136, 0.12153344208809136, 0.12071778140293637, 0.1199021207177814, 0.11745513866231648, 0.1166394779771615, 0.1166394779771615, 0.11582381729200653, 0.11582381729200653, 0.11500815660685156, 0.11092985318107668, 0.10929853181076672, 0.10929853181076672, 0.1068515497553018, 0.1068515497553018, 0.10358890701468189, 0.10114192495921696, 0.09461663947797716, 0.0]
rnn_neg = [0.0, 0.09654631083202511, 0.10282574568288853, 0.10675039246467818, 0.10989010989010989, 0.10989010989010989, 0.11538461538461539, 0.11930926216640503, 0.12166405023547881, 0.12166405023547881, 0.12244897959183673, 0.12480376766091052, 0.1271585557299843, 0.12951334379905807, 0.13186813186813187, 0.13186813186813187, 0.1326530612244898, 0.13343799058084774, 0.13422291993720564, 0.13657770800627944, 0.13736263736263737, 0.13814756671899528, 0.1389324960753532, 0.13971742543171115, 0.14207221350078492, 0.14207221350078492, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14364207221350078, 0.14442700156985872, 0.14521193092621665, 0.14521193092621665, 0.14521193092621665, 0.14521193092621665, 0.14599686028257458, 0.14678178963893249, 0.14678178963893249, 0.14756671899529042, 0.14835164835164835, 0.14835164835164835, 0.14835164835164835, 0.14835164835164835, 0.14913657770800628, 0.15070643642072212, 0.15306122448979592, 0.15384615384615385, 0.15384615384615385, 0.15384615384615385, 0.1546310832025118, 0.1554160125588697, 0.1554160125588697, 0.15698587127158556, 0.15855572998430142, 0.15934065934065933, 0.16012558869701726, 0.1609105180533752, 0.16169544740973313, 0.16169544740973313, 0.16248037676609106, 0.16248037676609106, 0.1640502354788069, 0.1640502354788069, 0.16483516483516483, 0.16483516483516483, 0.16562009419152277, 0.16562009419152277, 0.16718995290423863, 0.16718995290423863, 0.16797488226059654, 0.16797488226059654, 0.16875981161695447, 0.1695447409733124, 0.17111459968602827, 0.17189952904238617, 0.1726844583987441, 0.17346938775510204, 0.1750392464678179, 0.1750392464678179, 0.1750392464678179, 0.17739403453689168, 0.1781789638932496, 0.1781789638932496, 0.1781789638932496, 0.1781789638932496, 0.1781789638932496, 0.18053375196232338, 0.1836734693877551, 0.18524332810047095, 0.18681318681318682, 0.18995290423861852, 0.19387755102040816, 0.19623233908948196, 0.20094191522762953, 0.20486656200941916, 0.20800627943485087, 0.2119309262166405, 0.21978021978021978, 0.22919937205651492, 0.239403453689168, 1.0]
rnn_parm = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]     

# NN
# nn_pos = [1.0, 1.0, 1.0, 1.0, 0.99836867862969, 0.9902120717781403, 0.9861337683523654, 0.9861337683523654, 0.9853181076672104, 0.9820554649265906, 0.9804241435562806, 0.9771615008156607, 0.9755301794453507, 0.9730831973898858, 0.9698205546492659, 0.9690048939641109, 0.964110929853181, 0.9616639477977161, 0.9600326264274062, 0.9584013050570962, 0.9575856443719413, 0.9543230016313213, 0.9502446982055465, 0.9494290375203915, 0.9486133768352365, 0.9445350734094616, 0.9420880913539967, 0.9396411092985318, 0.9388254486133768, 0.9363784665579119, 0.932300163132137, 0.9282218597063622, 0.9241435562805873, 0.9233278955954323, 0.9216965742251223, 0.9184339314845025, 0.9168026101141925, 0.9127243066884176, 0.9110929853181077, 0.9070146818923328, 0.9070146818923328, 0.9053833605220228, 0.9029363784665579, 0.901305057096248, 0.899673735725938, 0.898042414355628, 0.8955954323001631, 0.8915171288743883, 0.8907014681892332, 0.8858075040783034, 0.8849918433931484, 0.8841761827079935, 0.8833605220228385, 0.8809135399673735, 0.8784665579119086, 0.8752039151712887, 0.8727569331158238, 0.8703099510603589, 0.8694942903752039, 0.865415986949429, 0.8629690048939641, 0.8605220228384992, 0.8588907014681892, 0.8572593800978793, 0.8548123980424144, 0.8531810766721044, 0.8499184339314845, 0.8458401305057096, 0.8417618270799347, 0.8368678629690048, 0.8344208809135399, 0.831973898858075, 0.8303425774877651, 0.8254486133768353, 0.8213703099510603, 0.8181076672104405, 0.8148450244698205, 0.8083197389885808, 0.8050570962479608, 0.8034257748776509, 0.8001631321370309, 0.7952691680261011, 0.7911908646003263, 0.7830342577487766, 0.7789559543230016, 0.7691680261011419, 0.7610114192495921, 0.7512234910277324, 0.7406199021207178, 0.7373572593800979, 0.7210440456769984, 0.7014681892332789, 0.6745513866231647, 0.6541598694942904, 0.634584013050571, 0.5921696574225123, 0.5415986949429038, 0.466557911908646, 0.3401305057096248, 0.0, 0.0]
# nn_neg = [0.0, 0.0, 0.0, 0.0, 0.0007849293563579278, 0.00706436420722135, 0.01020408163265306, 0.013343799058084773, 0.015698587127158554, 0.01726844583987441, 0.01805337519623234, 0.019623233908948195, 0.022762951334379906, 0.02511773940345369, 0.027472527472527472, 0.029827315541601257, 0.03296703296703297, 0.03453689167974882, 0.03453689167974882, 0.03532182103610675, 0.03610675039246468, 0.03924646781789639, 0.04081632653061224, 0.04317111459968603, 0.04552590266875981, 0.04631083202511774, 0.04709576138147567, 0.04866562009419152, 0.04945054945054945, 0.05180533751962323, 0.05416012558869702, 0.05572998430141287, 0.05886970172684458, 0.06357927786499215, 0.06828885400313972, 0.07221350078492936, 0.07221350078492936, 0.07692307692307693, 0.08006279434850863, 0.08163265306122448, 0.0847723704866562, 0.08634222919937205, 0.08791208791208792, 0.08869701726844584, 0.08948194662480377, 0.09105180533751962, 0.09497645211930926, 0.09576138147566719, 0.09811616954474098, 0.10125588697017268, 0.10361067503924647, 0.1043956043956044, 0.10675039246467818, 0.10910518053375197, 0.11067503924646782, 0.11224489795918367, 0.11459968602825746, 0.1185243328100471, 0.12009419152276295, 0.12244897959183673, 0.12637362637362637, 0.12872841444270017, 0.13422291993720564, 0.13736263736263737, 0.13971742543171115, 0.14285714285714285, 0.14442700156985872, 0.14599686028257458, 0.15384615384615385, 0.1577708006279435, 0.1609105180533752, 0.16326530612244897, 0.1695447409733124, 0.1726844583987441, 0.17346938775510204, 0.17660910518053374, 0.18210361067503925, 0.19073783359497645, 0.19152276295133439, 0.19858712715855573, 0.20408163265306123, 0.206436420722135, 0.20957613814756673, 0.2150706436420722, 0.2205651491365777, 0.2237048665620094, 0.23390894819466249, 0.24175824175824176, 0.25353218210361067, 0.2668759811616955, 0.27864992150706436, 0.2904238618524333, 0.3053375196232339, 0.3202511773940345, 0.347723704866562, 0.3791208791208791, 0.41915227629513346, 0.49058084772370486, 0.6098901098901099, 1.0, 1.0]
# nn_parm = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]     

nn_pos = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2838499184339315, 0.27161500815660683, 0.2634584013050571, 0.25122349102773245, 0.24877650897226752, 0.2463295269168026, 0.24388254486133767, 0.24225122349102773, 0.24143556280587275, 0.2398042414355628, 0.2365415986949429, 0.23409461663947798, 0.23164763458401305, 0.23083197389885807, 0.22920065252854813, 0.22920065252854813, 0.2275693311582382, 0.22512234910277323, 0.22512234910277323, 0.22430668841761828, 0.22430668841761828, 0.2234910277324633, 0.2226753670473083, 0.22104404567699837, 0.22022838499184338, 0.22022838499184338, 0.22022838499184338, 0.21941272430668843, 0.21859706362153344, 0.21778140293637846, 0.21778140293637846, 0.2169657422512235, 0.21533442088091354, 0.21451876019575855, 0.21451876019575855, 0.2137030995106036, 0.2137030995106036, 0.2128874388254486, 0.21207177814029363, 0.21207177814029363, 0.21125611745513867, 0.20880913539967375, 0.20799347471451876, 0.20799347471451876, 0.20717781402936378, 0.20636215334420882, 0.20636215334420882, 0.20636215334420882, 0.20554649265905384, 0.20228384991843393, 0.199836867862969, 0.19820554649265906, 0.19738988580750408, 0.19494290375203915, 0.19168026101141925, 0.19086460032626426, 0.1900489396411093, 0.18270799347471453, 0.17944535073409462, 0.1769983686786297, 0.16884176182707994, 0.1598694942903752, 0.14518760195758565, 0.13213703099510604, 0.1264274061990212, 0.12234910277324633, 0.11908646003262642, 0.1166394779771615, 0.11500815660685156, 0.11092985318107668, 0.1068515497553018, 0.10522022838499184, 0.10195758564437195, 0.09951060358890701, 0.09706362153344208, 0.09543230016313213, 0.09380097879282219, 0.08890701468189233, 0.08809135399673736, 0.08727569331158239, 0.08319738988580751, 0.08156606851549755, 0.0799347471451876, 0.0734094616639478, 0.0701468189233279, 0.06606851549755302, 0.06035889070146819, 0.04241435562805873, 0.00897226753670473, 0.0]
nn_neg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1075353218210361, 0.11381475667189953, 0.12009419152276295, 0.12401883830455258, 0.12637362637362637, 0.12872841444270017, 0.12872841444270017, 0.12951334379905807, 0.13108320251177394, 0.1326530612244898, 0.13343799058084774, 0.13343799058084774, 0.13657770800627944, 0.13736263736263737, 0.13814756671899528, 0.1389324960753532, 0.1389324960753532, 0.14050235478806908, 0.14050235478806908, 0.141287284144427, 0.141287284144427, 0.14207221350078492, 0.14207221350078492, 0.14364207221350078, 0.14364207221350078, 0.14364207221350078, 0.14442700156985872, 0.14521193092621665, 0.14521193092621665, 0.14678178963893249, 0.14835164835164835, 0.14835164835164835, 0.14835164835164835, 0.14913657770800628, 0.14992150706436422, 0.15149136577708006, 0.15149136577708006, 0.15306122448979592, 0.15384615384615385, 0.15384615384615385, 0.1554160125588697, 0.15698587127158556, 0.15698587127158556, 0.15855572998430142, 0.15855572998430142, 0.15855572998430142, 0.16012558869701726, 0.16012558869701726, 0.1609105180533752, 0.16326530612244897, 0.16718995290423863, 0.1695447409733124, 0.17111459968602827, 0.17425431711145997, 0.17660910518053374, 0.17896389324960754, 0.1836734693877551, 0.18681318681318682, 0.18838304552590268, 0.19387755102040816, 0.20094191522762953, 0.2150706436420722, 0.23861852433281006, 0.26373626373626374, 0.28649921507064363, 0.29984301412872844, 0.3076923076923077, 0.31240188383045525, 0.3163265306122449, 0.3218210361067504, 0.32731554160125587, 0.33359497645211933, 0.33751962323390894, 0.34222919937205654, 0.34850863422291994, 0.3540031397174254, 0.35949764521193095, 0.37127158555729983, 0.3751962323390895, 0.3814756671899529, 0.38618524332810045, 0.3963893249607535, 0.4089481946624804, 0.41601255886970173, 0.43328100470957615, 0.4489795918367347, 0.47017268445839877, 0.521978021978022, 0.7802197802197802, 1.0]
nn_parm = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

# LSTM
lstm_pos = [1.0, 0.4429037520391517, 0.3556280587275693, 0.32218597063621535, 0.29200652528548127, 0.2740619902120718, 0.26101141924959215, 0.2463295269168026, 0.23491027732463296, 0.22838499184339314, 0.2267536704730832, 0.22104404567699837, 0.21533442088091354, 0.20636215334420882, 0.19902120717781402, 0.19412724306688417, 0.19249592169657423, 0.1900489396411093, 0.18515497553017946, 0.1802610114192496, 0.1769983686786297, 0.17047308319738988, 0.16721044045676997, 0.1639477977161501, 0.16068515497553018, 0.1598694942903752, 0.1566068515497553, 0.15089722675367048, 0.14926590538336051, 0.14845024469820556, 0.14763458401305057, 0.1468189233278956, 0.14437194127243066, 0.14274061990212072, 0.14192495921696574, 0.14110929853181076, 0.1362153344208809, 0.13458401305057097, 0.132952691680261, 0.13213703099510604, 0.13050570962479607, 0.12887438825448613, 0.12479608482871125, 0.12234910277324633, 0.12071778140293637, 0.11745513866231648, 0.11745513866231648, 0.1166394779771615, 0.1133768352365416, 0.11092985318107668, 0.11011419249592169, 0.10766721044045677, 0.10766721044045677, 0.10766721044045677, 0.10766721044045677, 0.10603588907014681, 0.10522022838499184, 0.10522022838499184, 0.10522022838499184, 0.10522022838499184, 0.10440456769983687, 0.10114192495921696, 0.09787928221859707, 0.09787928221859707, 0.09706362153344208, 0.09624796084828711, 0.09380097879282219, 0.09216965742251224, 0.09135399673735727, 0.08972267536704731, 0.0864600326264274, 0.08401305057096248, 0.08238172920065252, 0.0799347471451876, 0.07830342577487764, 0.07748776508972267, 0.0766721044045677, 0.0766721044045677, 0.0766721044045677, 0.07585644371941272, 0.0734094616639478, 0.07259380097879282, 0.07259380097879282, 0.07177814029363784, 0.07096247960848287, 0.06933115823817292, 0.06606851549755302, 0.06443719412724307, 0.06199021207177814, 0.06035889070146819, 0.05954323001631321, 0.05464926590538336, 0.05464926590538336, 0.049755301794453505, 0.04404567699836868, 0.041598694942903754, 0.03915171288743882, 0.03181076672104405, 0.02936378466557912, 0.01794453507340946, 0.0]
lstm_neg = [0.0, 0.029042386185243328, 0.03532182103610675, 0.041601255886970175, 0.04552590266875981, 0.05180533751962323, 0.054945054945054944, 0.05572998430141287, 0.05886970172684458, 0.05886970172684458, 0.059654631083202514, 0.06357927786499215, 0.06671899529042387, 0.06828885400313972, 0.07221350078492936, 0.07299843014128729, 0.07378335949764521, 0.07927786499215071, 0.08241758241758242, 0.0847723704866562, 0.08634222919937205, 0.08712715855572999, 0.09183673469387756, 0.09262166405023547, 0.09340659340659341, 0.09497645211930926, 0.09576138147566719, 0.09811616954474098, 0.10047095761381476, 0.10361067503924647, 0.10518053375196232, 0.1075353218210361, 0.10910518053375197, 0.10989010989010989, 0.1130298273155416, 0.11538461538461539, 0.11616954474097331, 0.11695447409733124, 0.11773940345368916, 0.1185243328100471, 0.11930926216640503, 0.12166405023547881, 0.12244897959183673, 0.12558869701726844, 0.1271585557299843, 0.12794348508634223, 0.12951334379905807, 0.13108320251177394, 0.13422291993720564, 0.1357927786499215, 0.1357927786499215, 0.13971742543171115, 0.14285714285714285, 0.14442700156985872, 0.14521193092621665, 0.14756671899529042, 0.15070643642072212, 0.15149136577708006, 0.15384615384615385, 0.1554160125588697, 0.15698587127158556, 0.15855572998430142, 0.15934065934065933, 0.1609105180533752, 0.16483516483516483, 0.16718995290423863, 0.16797488226059654, 0.16875981161695447, 0.17111459968602827, 0.17582417582417584, 0.17660910518053374, 0.17896389324960754, 0.18524332810047095, 0.18995290423861852, 0.19230769230769232, 0.19623233908948196, 0.2001569858712716, 0.206436420722135, 0.20722135007849293, 0.21036106750392464, 0.21585557299843014, 0.21821036106750394, 0.2237048665620094, 0.22684458398744112, 0.22919937205651492, 0.23233908948194662, 0.23861852433281006, 0.2425431711145997, 0.24725274725274726, 0.25353218210361067, 0.2629513343799058, 0.271585557299843, 0.28414442700156983, 0.3029827315541601, 0.315541601255887, 0.33124018838304553, 0.3563579277864992, 0.3956043956043956, 0.44427001569858715, 0.5580847723704867, 1.0]
lstm_parm = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

# Perceptron
per_pos = [1.0, 0.935562805872757, 0.8923327895595432, 0.8621533442088092, 0.8287112561174551, 0.8115823817292006, 0.7862969004893964, 0.763458401305057, 0.7471451876019576, 0.7251223491027733, 0.7063621533442088, 0.6835236541598695, 0.6606851549755302, 0.6411092985318108, 0.6223491027732463, 0.6011419249592169, 0.5815660685154975, 0.5636215334420881, 0.5513866231647635, 0.5350734094616639, 0.5212071778140294, 0.5048939641109299, 0.48613376835236544, 0.4706362153344209, 0.4551386623164764, 0.4461663947797716, 0.4355628058727569, 0.4233278955954323, 0.4159869494290375, 0.400489396411093, 0.3907014681892333, 0.3768352365415987, 0.36541598694942906, 0.35073409461663946, 0.33931484502446985, 0.3278955954323002, 0.3164763458401305, 0.30424143556280586, 0.2895595432300163, 0.27895595432300163, 0.27161500815660683, 0.25856443719412725, 0.2495921696574225, 0.24225122349102773, 0.233278955954323, 0.2234910277324633, 0.21859706362153344, 0.21125611745513867, 0.20554649265905384, 0.19820554649265906, 0.1933115823817292, 0.1835236541598695, 0.17128874388254486, 0.16721044045676997, 0.1598694942903752, 0.14926590538336051, 0.13947797716150082, 0.1370309951060359, 0.12887438825448613, 0.12071778140293637, 0.11256117455138662, 0.10929853181076672, 0.10114192495921696, 0.09624796084828711, 0.09216965742251224, 0.08890701468189233, 0.08401305057096248, 0.08238172920065252, 0.08156606851549755, 0.08075040783034258, 0.07422512234910278, 0.06933115823817292, 0.0636215334420881, 0.06280587275693311, 0.06035889070146819, 0.05872756933115824, 0.05464926590538336, 0.05301794453507341, 0.048939641109298535, 0.04404567699836868, 0.04241435562805873, 0.0399673735725938, 0.037520391517128875, 0.03588907014681892, 0.03507340946166395, 0.033442088091353996, 0.03262642740619902, 0.03181076672104405, 0.028548123980424143, 0.02773246329526917, 0.024469820554649267, 0.021207177814029365, 0.01957585644371941, 0.017128874388254486, 0.015497553017944535, 0.013050570962479609, 0.008156606851549755, 0.0065252854812398045, 0.004078303425774877, 0.0032626427406199023, 0.0]
per_neg = [0.0, 0.0007849293563579278, 0.0031397174254317113, 0.003924646781789639, 0.004709576138147566, 0.005494505494505495, 0.005494505494505495, 0.005494505494505495, 0.005494505494505495, 0.007849293563579277, 0.01098901098901099, 0.014913657770800628, 0.01726844583987441, 0.019623233908948195, 0.02119309262166405, 0.02511773940345369, 0.0282574568288854, 0.029827315541601257, 0.03139717425431711, 0.03218210361067504, 0.033751962323390894, 0.03532182103610675, 0.040031397174254316, 0.0423861852433281, 0.04631083202511774, 0.04945054945054945, 0.05337519623233909, 0.05886970172684458, 0.065149136577708, 0.06593406593406594, 0.07142857142857142, 0.07378335949764521, 0.07692307692307693, 0.08084772370486656, 0.08948194662480377, 0.09811616954474098, 0.10204081632653061, 0.10596546310832025, 0.10989010989010989, 0.1130298273155416, 0.1185243328100471, 0.12244897959183673, 0.12637362637362637, 0.13186813186813187, 0.1357927786499215, 0.14678178963893249, 0.15149136577708006, 0.1546310832025118, 0.1609105180533752, 0.1664050235478807, 0.17425431711145997, 0.18524332810047095, 0.19230769230769232, 0.19544740973312402, 0.2087912087912088, 0.217425431711146, 0.22919937205651492, 0.23783359497645212, 0.24411302982731553, 0.2543171114599686, 0.26609105180533754, 0.2755102040816326, 0.2857142857142857, 0.2943485086342229, 0.2990580847723705, 0.3116169544740973, 0.3218210361067504, 0.33751962323390894, 0.35086342229199374, 0.3642072213500785, 0.3728414442700157, 0.38304552590266877, 0.4003139717425432, 0.41836734693877553, 0.434850863422292, 0.45290423861852436, 0.46781789638932497, 0.47959183673469385, 0.49686028257456827, 0.5125588697017268, 0.5274725274725275, 0.543171114599686, 0.5612244897959183, 0.576138147566719, 0.5910518053375197, 0.6036106750392465, 0.6224489795918368, 0.6357927786499215, 0.6585557299843015, 0.6766091051805337, 0.7001569858712716, 0.7252747252747253, 0.7456828885400314, 0.7660910518053375, 0.7880690737833596, 0.8069073783359497, 0.82574568288854, 0.8532182103610675, 0.8877551020408163, 0.9246467817896389, 1.0]
per_parm = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]


seriesFPRs = [per_pos, nn_pos, rnn_pos, lstm_pos]
seriesFNRs= [per_neg, nn_neg, rnn_neg, lstm_neg]
seriesLabels = ["Perceptron", "Feedforward NN", "Recurrent NN", "LSTM"]

data_plotter.plot_rocs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-SMSSpamROCs")
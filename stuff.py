def wordchunks(text, token_per_chunk):
    token = text.split()
    token_count = len(token)
    full_parts = token_count // token_per_chunk
    chunks = [token[i * token_per_chunk: (i + 1) * token_per_chunk] for i in range(0, full_parts)]
    if token_count % token_per_chunk != 0:
        chunks.append(token[full_parts * token_per_chunk:])
    return list(map(lambda x: ' '.join(x), chunks))

def last_wordchunk(text, chunk_length):
    return ' '.join(text.split()[-chunk_length:])

def first_wordchunk(text, chunk_length):
    return ' '.join(text.split()[:chunk_length])

text = "Indigenous group to lawmaker: Correct your faults against refugees The Philippine Task Force for Indigenous Peoples’ Rights calls for accountability in light of the recent violent incident that hurt indigenous Lumad evacuees and advocates in Mindanao Militarization of communities in the Philippine countryside has intensified due to the government’s counter-insurgency operations against revolutionary groups as part of Operation Bayanihan. But clearly, the presence of the military in indigenous communities is also meant to stifle and repress legitimate dissent against unwanted projects that are destructive to their land and natural resources . The military are being used to secure the interests of giant companies implementing large mining , energy, logging , plantations, and tourism projects in indigenous peoples’ ancestral domains. Having participated in several solidarity and fact-finding missions in evacuation centers where indigenous peoples have sought sanctuary, the Philippine Task Force for Indigenous Peoples’ Rights (TFIP) has witnessed the great difficulties that indigenous peoples experience in leaving their homes and risking hunger and illness in the evacuation centers. Yet, they chose to evacuate in their desire for safety, to preserve their lives from harassment by the military and paramilitary forces. In their communities, they feel defenseless against the high-powered rifles of the soldiers. The explosion of bombs and the roar of gunfire are much more frightening than the noisy traffic jams and the usual hustle and bustle of the city. They did not evacuate merely to get away from scary gun-toting soldiers. They went to the city to be closer to and to reach out to government offices, churches, human rights defenders, advocates and supporters from whom they could seek support. But being in a different environment far away from their homes has had dire impacts on their health, nutrition, sanitation and living conditions . The role of church organizations, human rights defenders, and advocacy groups in such crisis situations has been admirable. They provide sanctuary, legal assistance, and humanitarian services to the refugees. Human rights organizations and defenders document the events and circumstances leading to the evacuation, including the people’s demands to the government. They also assist by relaying the information to concerned government agencies and the wider public through various media. (READ: UN Special Rapporteur to PH: Defend Lumads from militarization ) DID SHE INSULT THE LUMAD? North Cotabato 2nd District Representative Nancy Catamco, who is an indigenous person, denies the allegation. Photo by Kilab Multimedia Congresswoman Nancy Catamco seems unable to comprehend the dire situation of the more than 700 indigenous Manobo refugees from Kapalong and Talaingod, Davao who have been given sanctuary by the United Council of Churches in the Philippines (UCCP) in Haran Mission House in Davao City for several months now. She was first invited to visit the evacuation site by the Save Our Schools Network, a network of organizations aiming to save indigenous schools from attacks by the military, Department of Education (DEPED), and local government units, which have vilified these schools and teachers as New People's Army (NPA) supporters. Many of the refugees are women and children who were forced to stop schooling because of militarization and the attacks against their community schools. The evacuation center also serves as a temporary school for these children while their community schools are currently suspended and ordered closed by DEPED. Other Stories The UN Special Rapporteur on the Rights of Indigenous Peoples thinks the proposed Bangsamoro law falls short in meeting the minimum international standards for the survival, dignity, and well being of indigenous peoples Lumad leaders are worried the exclusion of their rights to ancestral lands may result in the loss of their agricultural resources The government's intensified campaign against communist rebels makes indigenous peoples vulnerable to violence; the specter of death underpins their lives like an active fault line Congresswoman Catamco was expected, at the very least, to listen to the refugees’ pleas and demands, as she claims to stand for the interests of indigenous peoples, being the head of the Committee on National Cultural Communities of the House of Representatives. It was hoped that she would take steps to respond to the basic demand of the indigenous refugees to pull out the military from their communities so that they could go back home to resume their peaceful lives and livelihoods in their communities. But the indigenous peoples were shocked by the congresswoman’s words and actions in the evacuation center. First, she acted in bad faith by bringing with her Philippine Army's 1003rd Brigade commander Colonel Harold Cabreros and Brigadier General Alexander Baluta of the Eastern Mindanao Command, who represent the very threat that the refugees are fleeing from. (READ: Earlier, she proposed a meeting and dialogue between the refugees and government agencies including fellow lawmakers. The indigenous peoples agreed to this meeting on the condition that she would not bring with her representatives of the military. Second, upon arriving at the evacuation center, she was emotional and angrily scolded the indigenous peoples, the UCCP, and the human rights defenders of Davao City. She berated the indigenous peoples, suggested that they stink and ordered them to go back to their communities as she said that they are being treated inhumanely in the evacuation center. (READ: Did lawmaker call Lumad evacuees ‘stinky’? ) Without properly listening to the account of indigenous peoples regarding the events and situation that led them to evacuate, Congresswoman Catamco judged their dismal situation in the evacuation center as inhumane treatment by their hosts. She further maligned the UCCP and the human rights organizations by falsely accusing them of abducting the indigenous peoples and putting them in a concentration camp. CLASH. Two cops are hurt while at least 17 Lumad leaders and their supporters were injured in a clash that happened inside the United Church of Christ in the Philippines (UCCP) Haran compound in Davao City. Photo by Karlos Manlupig Worst, she sent in the police and paramilitary forces on July 23, who raided and broke down the gate of the UCCP Haran compound and destroyed the temporary shelters of the refugees. At least  17 refugees and UCCP pastors were hurt when the truncheon-wielding cops attempted to evict them. (READ: Lumad evacuees, activists clash with cops in Davao ) We condemn this violent incident and the brutal use of force by the police against the refugees upon the instigation of Congresswoman Catamco. (READ: Duterte: Lawmaker to blame for clash in Lumad evacuees' site ) In doing all these, Congresswoman Catamco exposed her ignorance and discriminatory attitude against indigenous peoples. Her conduct is uncalled for, malicious and violates the ethics of a legislator and a parliamentarian. She abused her position as an elected legislator and as the appointed chairperson of the Committee on National Cultural Communities of the House of Representatives. She must apologize for her insults and faults against the Manobo people and rectify her mistakes. She should take efforts to impartially investigate the issues raised by the indigenous peoples and address their demands in order to facilitate their safe return to their communities free from militarization and human rights violations. If she can’t do this, we believe that she is unfit to be a congresswoman, much less the chairperson of the House committee that is tasked to uphold the rights and welfare of indigenous peoples. – Jill Cariño is an indigenous Ibaloi from Baguio and Executive Director of the Philippine Task Force for Indigenous Peoples’ Rights (TFIP), a national network of NGOs working to advance the rights of indigenous peoples."
for t in wordchunks(text, 192):
    print(t)

print()
print(last_wordchunk(text, 192))
print()
print(first_wordchunk(text, 192))
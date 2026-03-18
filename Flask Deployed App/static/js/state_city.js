var state_arr = new Array("Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal");

var s_a = new Array();
s_a[0] = "";
s_a[1] = "Visakhapatnam|Vijayawada|Guntur|Nellore|Kurnool|Kadapa|Rajahmundry|Kakinada|Tirupati|Anantapur";
s_a[2] = "Itanagar|Naharlagun|Pasighat|Roing|Tezu|Namsai|Tawang|Bomdila|Ziro|Along";
s_a[3] = "Guwahati|Silchar|Dibrugarh|Jorhat|Nagaon|Tinsukia|Tezpur|Karimganj|Goalpara|Diphu";
s_a[4] = "Patna|Gaya|Bhagalpur|Muzaffarpur|Purnia|Darbhanga|Bihar Sharif|Arrah|Begusarai|Katihar";
s_a[5] = "Raipur|Bhilai|Bilaspur|Korba|Rajnandgaon|Raigarh|Jagdalpur|Dhamtari|Chirmiri|Durg";
s_a[6] = "Panaji|Margao|Vasco da Gama|Mapusa|Ponda|Bicholim|Curchorem|Cuncolim|Canacona|Pernem";
s_a[7] = "Ahmedabad|Surat|Vadodara|Rajkot|Bhavnagar|Jamnagar|Gandhinagar|Junagadh|Anand|Nadiad";
s_a[8] = "Faridabad|Gurugram|Panipat|Ambala|Yamunanagar|Rohtak|Hisar|Karnal|Sonipat|Panchkula";
s_a[9] = "Shimla|Dharamshala|Solan|Mandi|Kullu|Manali|Nahan|Bilaspur|Hamirpur|Una";
s_a[10] = "Ranchi|Jamshedpur|Dhanbad|Bokaro|Hazaribagh|Deoghar|Giridih|Ramgarh|Medininagar|Chatra";
s_a[11] = "Bangalore|Mysore|Hubli|Mangalore|Belgaum|Gulbarga|Davanagere|Bellary|Bijapur|Shimoga";
s_a[12] = "Thiruvananthapuram|Kochi|Kozhikode|Thrissur|Kollam|Palakkad|Alappuzha|Kannur|Kottayam|Malappuram";
s_a[13] = "Bhopal|Indore|Jabalpur|Gwalior|Ujjain|Sagar|Dewas|Satna|Ratlam|Rewa";
s_a[14] = "Mumbai|Pune|Nagpur|Thane|Nashik|Aurangabad|Solapur|Kolhapur|Amravati|Navi Mumbai";
s_a[15] = "Imphal|Bishnupur|Thoubal|Churachandpur|Senapati|Ukhrul|Chandel|Tamenglong|Jiribam|Kangpokpi";
s_a[16] = "Shillong|Tura|Nongstoin|Jowai|Baghmara|Resubelpara|Williamnagar|Nongpoh|Mairang|Cherrapunji";
s_a[17] = "Aizawl|Lunglei|Saiha|Champhai|Kolasib|Serchhip|Lawngtlai|Mamit|Khawzawl|Hnahthial";
s_a[18] = "Kohima|Dimapur|Mokokchung|Tuensang|Wokha|Zunheboto|Phek|Mon|Peren|Kiphire";
s_a[19] = "Bhubaneswar|Cuttack|Rourkela|Berhampur|Sambalpur|Puri|Balasore|Bhadrak|Baripada|Jharsuguda";
s_a[20] = "Ludhiana|Amritsar|Jalandhar|Patiala|Bathinda|Mohali|Pathankot|Hoshiarpur|Batala|Moga";
s_a[21] = "Jaipur|Jodhpur|Kota|Bikaner|Ajmer|Udaipur|Bhilwara|Alwar|Sikar|Sri Ganganagar";
s_a[22] = "Gangtok|Namchi|Gyalshing|Mangan|Rangpo|Singtam|Jorethang|Nayabazar|Ravangla|Soreng";
s_a[23] = "Chennai|Coimbatore|Madurai|Tiruchirappalli|Salem|Tirunelveli|Tiruppur|Vellore|Erode|Thoothukkudi";
s_a[24] = "Hyderabad|Warangal|Nizamabad|Karimnagar|Ramagundam|Khammam|Mahbubnagar|Nalgonda|Adilabad|Suryapet";
s_a[25] = "Agartala|Udaipur|Dharmanagar|Pratapgarh|Belonia|Kailasahar|Khowai|Teliamura|Mohanpur|Melaghar";
s_a[26] = "Lucknow|Kanpur|Varanasi|Agra|Prayagraj|Meerut|Bareilly|Aligarh|Moradabad|Saharanpur";
s_a[27] = "Dehradun|Haridwar|Roorkee|Haldwani|Rudrapur|Kashipur|Rishikesh|Pithoragarh|Ramnagar|Roorkee";
s_a[28] = "Kolkata|Howrah|Durgapur|Asansol|Siliguri|Maheshtala|Rajpur Sonarpur|South Dumdum|Gopalpur|Bhatpara";

function print_state(state_id){
    var option_str = document.getElementById(state_id);
    option_str.length=0;
    option_str.options[0] = new Option('Select State','');
    option_str.selectedIndex = 0;
    for (var i=0; i<state_arr.length; i++) {
        option_str.options[option_str.length] = new Option(state_arr[i],state_arr[i]);
    }
}

function print_city(city_id, state_index){
    var option_str = document.getElementById(city_id);
    option_str.length=0;
    option_str.options[0] = new Option('Select City','');
    option_str.selectedIndex = 0;
    var state_arr_index = state_index;
    if (state_arr_index>0) {
        var city_arr = s_a[state_arr_index].split("|");
        for (var i=0; i<city_arr.length; i++) {
            option_str.options[option_str.length] = new Option(city_arr[i],city_arr[i]);
        }
    }
}
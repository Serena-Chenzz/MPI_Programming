from mpi4py import MPI 
import json
import numpy as np
import re

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Here, we define some variables for all processors.
# These counters are used to count post number for each processor.
count_post_per_box = {}
count_post_per_row = {}
count_post_per_column = {}

#These variables are used by master node to sum up the post numbers.
sum_post_for_box_per_pro = None
sum_post_for_row_per_pro = None
sum_post_for_column_per_pro = None

#############------------------------------------------------Function Definition---------------------------------------------######################
# I first define some functions which will be used next.
# Get rid of all useless attributes in json object. Only the information of coordinates will be recorded
def get_filtered_data(json_obj):
    json_data = None
    
    if "doc" in json_obj:
        if "coordinates" in json_obj["doc"]:
            if "coordinates" in json_obj["doc"]["coordinates"]:
                json_data = {"coordinates": json_obj["doc"]["coordinates"]["coordinates"]}

    return json_data

# Check whether it is inside the whole range of interest
def check_whether_in_range(x_value, y_value, range_of_data):
    A1_xmin = range_of_data[0][0]
    A1_ymax = range_of_data[0][1]
    C4_xmax = range_of_data[1][0]
    C4_ymin = range_of_data[1][1]
    C3_xmin = range_of_data[2][0]
    C3_ymax = range_of_data[2][1]
    D5_xmax = range_of_data[3][0]
    D5_ymin = range_of_data[3][1]
   
    if x_value >= A1_xmin and x_value <= C4_xmax and y_value <= A1_ymax and y_value >= C4_ymin:
        return True
    elif x_value >= C3_xmin and x_value <= D5_xmax and y_value <= C3_ymax and y_value >= D5_ymin:
        return True
    else:
        return False

# Check which box the coordinate pair belongs to
def get_box_id_for_json_object(x_value, y_value, range_of_each_box):

    #If the coordinates fall on the overlapped edge, it belongs to the biggest box_id of them. For example, if the point falls on the common edge of A1 and A2,
    #it belongs to A2. If the point falls on the center point of A1, A2, B1, B2, it belongs to B2.
    #Record all possible box_ids
    ids = []
    for box in range_of_each_box:
        if (x_value >= box["xmin"] and x_value <= box["xmax"] and y_value >= box["ymin"] and y_value <= box["ymax"]):
            ids.append(box["id"])
    # Coordiantes on the overlapped edge belong to the last box_id
    # If it does not belong to any box, it will return None.
    try:
        return ids[-1]
    except IndexError:
        return None


#############------------------------------------------------Master node's work---------------------------------------------######################
# Master node reads 'melbgrid' file and broadcasts some range parameters so slavers don't have to open this file again
if rank == 0:
    grid_filename = "melbGrid.json"
    
    # range_of_each box is used to record xmin, xmax, ymin, ymax of each box
    range_of_each_box = []

    with open(grid_filename, "r", encoding="utf-8") as grid_file:
        whole_obj = json.load(grid_file)
        box_objs = whole_obj["features"]

        for box_obj in box_objs:

            # Record each box's vertexes.
            range_of_box = {"id": box_obj["properties"]["id"],
                            "xmin": box_obj["properties"]["xmin"],
                            "xmax": box_obj["properties"]["xmax"],
                            "ymin": box_obj["properties"]["ymin"],
                            "ymax": box_obj["properties"]["ymax"],
                            }
            range_of_each_box.append(range_of_box)

            # Fetch the vertex coordinates of A1, C4, C3, D5 since the union of rectangle(A1, C4) and rectangle(C4, D5) defines the whole range of interest
            # These coordinates will be broadcast to all other processors
            if box_obj["properties"]["id"] == "A1":
                A1_xmin = box_obj["properties"]["xmin"]
                A1_ymax = box_obj["properties"]["ymax"]
            elif box_obj["properties"]["id"] == "C4":
                C4_xmax = box_obj["properties"]["xmax"]
                C4_ymin = box_obj["properties"]["ymin"]
            elif box_obj["properties"]["id"] == "C3":
                C3_xmin = box_obj["properties"]["xmin"]
                C3_ymax = box_obj["properties"]["ymax"]
            elif box_obj["properties"]["id"] == "D5":
                D5_xmax = box_obj["properties"]["xmax"]
                D5_ymin = box_obj["properties"]["ymin"]

            # Initialize the counter dictionaries for each processor to record the post number:
            # For example, post numbers will be recorded like {"A1":0, "A2":0, "B1":0....} in each processor
            if box_obj["properties"]["id"] not in count_post_per_box:
                count_post_per_box[box_obj["properties"]["id"]] = 0

            if box_obj["properties"]["id"][0] not in count_post_per_row:
                count_post_per_row[box_obj["properties"]["id"][0]] = 0

            if box_obj["properties"]["id"][1] not in count_post_per_column:
                count_post_per_column[box_obj["properties"]["id"][1]] = 0

    # Range_of_data is to define the peripheral border of the whole range.
    range_of_data = [(A1_xmin,A1_ymax),(C4_xmax,C4_ymin),(C3_xmin,C3_ymax),(D5_xmax,D5_ymin)]


else:
    range_of_data = None
    range_of_each_box = None
    count_post_per_box = None
    count_post_per_row = None
    count_post_per_column = None

# Broadcast whole range of interest as well as range for each box    
range_of_data = comm.bcast(range_of_data)
range_of_each_box = comm.bcast(range_of_each_box)

# Broadcast dictionaries for processors to fill in. Dictionaries are initialized like this {"A1":0, "A2":0, "B1":0....}.
count_post_per_box = comm.bcast(count_post_per_box)
count_post_per_row = comm.bcast(count_post_per_row)
count_post_per_column = comm.bcast(count_post_per_column)


#############------------------------------------------------Parallel Computing Starts Here---------------------------------------------######################
data_origin_filename = "bigInstagram.json"

# Counter is used to record current line sequence number
counter = -1
# First, collect all the data within the range and check which boxes they belong to
data_of_interest = []
# Open the data file
with open(data_origin_filename, "r", encoding="utf-8") as data_origin_file:
    # Read and process the file line by line except for the first line and last line
    for line in data_origin_file:
        counter += 1
        # Check if it is the first line, if so, skip it.
        if counter == 0:
            continue
        # Also check if it is the last line, if so, skip it. For tinyInstagram and mediumInstagram file, the last line starts with ']}'
        if re.match(r'^]}.*', line):
            break
        
        # Rank 0 processes line 0, rank 1 processes line 1, rank 2 processes line 2....
        if counter % size == rank:
            json_obj = {}

            #Last line of data ends with '}]}' in bigInstagram file
            if re.match(r'.*}]}$', line):
                json_obj = json.loads(line.strip()[:-2])
            
            #Last line of data ends with '}' in tinyInstagram & mediumInstagram file
            elif re.match(r'.*}$', line):
                json_obj = json.loads(line.strip())

            else:
                # For other lines, get rid of the ',' at the end of the line
                json_obj = json.loads(line.strip()[:-1])

            #Check if this json object has 'coordinates' key
            json_data = get_filtered_data(json_obj)
            if json_data != None:
                x = json_data["coordinates"][1]
                y = json_data["coordinates"][0]
                
                # Check if this json object inside the range of interest
                if x != None and y != None and check_whether_in_range(x,y,range_of_data):
                    # Record the box_id for json object using dictionary
                    box_id = get_box_id_for_json_object(x,y,range_of_each_box)
                    if box_id != None:
                        dict_new = {"x_value":x, "y_value":y, "box_id": box_id}
                        data_of_interest.append(dict_new)

# Second, each processor calculates post number for each box, row, column using the key "box_id"
for obj_of_interest in data_of_interest:
    count_post_per_box[obj_of_interest["box_id"]] += 1
    count_post_per_row[obj_of_interest["box_id"][0]] += 1
    count_post_per_column[obj_of_interest["box_id"][1]] += 1


# Gather the results from all processors to master node
sum_post_for_box_per_pro = comm.gather(count_post_per_box,root = 0)
sum_post_for_row_per_pro = comm.gather(count_post_per_row,root = 0)
sum_post_for_column_per_pro = comm.gather(count_post_per_column, root = 0)

if rank == 0:
    
    # Handle the lists collected from all processors. Master node sums them up for each row, column and box.
    sum_post_for_box={}
    sum_post_for_row={}
    sum_post_for_column={}
    
    for each_item in sum_post_for_box_per_pro:
        for each_box in each_item:
            if each_box not in sum_post_for_box:
                sum_post_for_box[each_box] = 0
            sum_post_for_box[each_box] += each_item[each_box]

    for key in sum_post_for_box:
        if key[0] not in sum_post_for_row:
            sum_post_for_row[key[0]] = 0
        sum_post_for_row[key[0]] += sum_post_for_box[key]

    for key in sum_post_for_box:
        if key[1] not in sum_post_for_column:
            sum_post_for_column[key[1]] = 0
        sum_post_for_column[key[1]] += sum_post_for_box[key]


    # Sort the results and generate output
    print("Order the grid box based on the number of posts in each box")
    for post_pair in sorted(sum_post_for_box.items(), key = lambda x:x[1], reverse = True):
        print(post_pair[0] + " : " + str(post_pair[1]))

    print("\nOrder the grid rows based on the number of posts in each row")
    for post_pair in sorted(sum_post_for_row.items(), key = lambda x:x[1], reverse = True):
        print(post_pair[0] + " : " + str(post_pair[1]))

    print("\nOrder the grid columns based on the number of posts in each column")
    for post_pair in sorted(sum_post_for_column.items(), key = lambda x:x[1], reverse = True):
        print(post_pair[0] + " : " + str(post_pair[1]))


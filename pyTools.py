#!/usr/bin/env python
'''
Tools are taken from ASAPPyTools
https://github.com/NCAR/ASAPPyTools
which is no longer actively supported ...
'''

from collections import defaultdict

# Define the supported reduction operators
OPERATORS = ['sum', 'prod', 'max', 'min']

def create_comm(serial=False):
    """
    Depending on the argument given, it returns an instance of a 
    serial or parallel SimpleComm object.                                        
    """
    if type(serial) is not bool:
        raise TypeError('Serial parameter must be a bool')
    if serial:
        return SimpleComm()
    else:
        return SimpleCommMPI()

class SimpleComm(object):

    """
    Simple Communicator for serial operation.

    Attributes:
        _numpy: Reference to the Numpy module, if found
    """

    def __init__(self):
        # Try importing the Numpy module
        try:
            import numpy
        except:
            numpy = None
        # To the Numpy module, if found
        self._numpy = numpy

    def _is_ndarray(self, obj):
        if self._numpy:
            return isinstance(obj, self._numpy.ndarray)
        else:
            return False

    def get_size(self):
        #Get the integer number of ranks in this communicator.
        return 1

    def get_rank(self):
        #Get the integer rank ID of this MPI process in this communicator.
        return 0

    def is_manager(self):
        return self.get_rank() == 0

    def sync(self):
        return

    def allreduce(self, data, op):
        """
        Parameters:
            data: The data to be reduced
            op (str): A string identifier for a reduce operation (any string
                found in the OPERATORS list)

        Returns:
            The single value constituting the reduction of the input data.
            (The same value is returned on all ranks in this communicator.)
        """
        if isinstance(data, dict):
            totals = {}
            for k, v in data.items():
                totals[k] = SimpleComm.allreduce(self, v, op)
            return totals
        elif self._is_ndarray(data):
            return SimpleComm.allreduce(self, getattr(self._numpy, _OP_MAP[op]['np'])(data), op)
        elif hasattr(data, '__len__'):
            return SimpleComm.allreduce(self, eval(_OP_MAP[op]['py'])(data), op)
        else:
            return data
    def partition(self, data=None, func=None, involved=False, tag=0):
        """
        Partition and send data from the 'manager' rank to 'worker' ranks.

        By default, the data is partitioned using an "equal stride" across the
        data, with the stride equal to the number of ranks involved in the
        partitioning.  If a partition function is supplied via the `func`
        argument, then the data will be partitioned across the 'worker' ranks,
        giving each 'worker' rank a different part of the data according to
        the algorithm used by partition function supplied.

        If the `involved` argument is True, then a part of the data (as
        determined by the given partition function, if supplied) will be
        returned on the 'manager' rank.  Otherwise, ('involved' argument is
        False) the data will be partitioned only across the 'worker' ranks.

        This call must be made by all ranks.

        Keyword Arguments:
            data: The data to be partitioned across the ranks in the
                communicator.
            func: A PartitionFunction object/function that returns a part
                of the data given the index and assumed size of the partition.
            involved (bool): True if a part of the data should be given to the
                'manager' rank in addition to the 'worker' ranks. False
                otherwise.
            tag (int): A user-defined integer tag to uniquely specify this
                communication message.

        Returns:
            A (possibly partitioned) subset (i.e., part) of the data.  Depending
            on the PartitionFunction used (or if it is used at all), this method
            may return a different part on each rank.
        """
        op = func if func else lambda *x: x[0][x[1] :: x[2]]
        if involved:
            return op(data, 0, 1)
        else:
            return None

    def ration(self, data=None, tag=0):
        err_msg = 'Rationing cannot be used in serial operation'
        raise RuntimeError(err_msg)

    def collect(self, data=None, tag=0):
        err_msg = 'Collection cannot be used in serial operation'
        raise RuntimeError(err_msg)

class SimpleCommMPI(SimpleComm):

    """
    Simple Communicator using MPI.

    Attributes:
        PART_TAG: Partition Tag Identifier
        RATN_TAG: Ration Tag Identifier
        CLCT_TAG: Collect Tag Identifier
        REQ_TAG: Request Identifier
        MSG_TAG: Message Identifer
        ACK_TAG: Acknowledgement Identifier
        PYT_TAG: Python send/recv Identifier
        NPY_TAG: Numpy send/recv Identifier
        _mpi: A reference to the mpi4py.MPI module
        _comm: A reference to the mpi4py.MPI communicator
    """

    PART_TAG = 1  # Partition Tag Identifier
    RATN_TAG = 2  # Ration Tag Identifier
    CLCT_TAG = 3  # Collect Tag Identifier

    REQ_TAG = 1  # Request Identifier
    MSG_TAG = 2  # Message Identifier
    ACK_TAG = 3  # Acknowledgement Identifier
    PYT_TAG = 4  # Python Data send/recv Identifier
    NPY_TAG = 5  # Numpy NDArray send/recv Identifier

    def __init__(self):
        # Call the base class constructor
        super(SimpleCommMPI, self).__init__()

        # Try importing the MPI4Py MPI module
        try:
            from mpi4py import MPI
        except:
            err_msg = 'MPI could not be found.'
            raise ImportError(err_msg)

        # Hold on to the MPI module
        self._mpi = MPI

        # The MPI communicator (by default, COMM_WORLD)
        self._comm = self._mpi.COMM_WORLD

    def __del__(self):
        if self._comm != self._mpi.COMM_WORLD:
            self._comm.Free()

    def _is_bufferable(self, obj):
        """
        Check if the data is bufferable or not
        """
        if self._is_ndarray(obj):
            if hasattr(self._mpi, '_typedict_c'):
                return obj.dtype.char in self._mpi._typedict_c
            elif hasattr(self._mpi, '__CTypeDict__'):
                return obj.dtype.char in self._mpi.__CTypeDict__ and obj.dtype.char != 'c'
            else:
                return False
        else:
            return False

    def get_size(self):
        #Get the integer number of ranks in this communicator.
        return self._comm.Get_size()

    def get_rank(self):
        return self._comm.Get_rank()

    def sync(self):
        self._comm.Barrier()

    def allreduce(self, data, op):
        """
        Perform an MPI AllReduction operation.

              Returns:
            The single value constituting the reduction of the input data.
            (The same value is returned on all ranks in this communicator.)
        """
        if isinstance(data, dict):
            all_list = self._comm.gather(SimpleComm.allreduce(self, data, op))
            if self.is_manager():
                all_dict = defaultdict(list)
                for d in all_list:
                    for k, v in d.items():
                        all_dict[k].append(v)
                result = {}
                for k, v in all_dict.items():
                    result[k] = SimpleComm.allreduce(self, v, op)
                return self._comm.bcast(result)
            else:
                return self._comm.bcast(None)
        else:
            return self._comm.allreduce(
                SimpleComm.allreduce(self, data, op),
                op=getattr(self._mpi, _OP_MAP[op]['mpi']),
            )

    def _tag_offset(self, method, message, user):
        """
        Method to generate the tag for a given MPI message

        Parameters:
            method (int): One of PART_TAG, RATN_TAG, CLCT_TAG
            message (int):  One of REQ_TAG, MSG_TAG, ACK_TAG, PYT_TAG, NPY_TAG
            user (int): A user-defined integer tag

        Returns:
            int: A new tag uniquely combining all of the method, message, and
                user tags together
        """
        return 100 * user + 10 * method + message

    def partition(self, data=None, func=None, involved=False, tag=0):
        """
        Partition and send data from the 'manager' rank to 'worker' ranks.

        By default, the data is partitioned using an "equal stride" across the
        data, with the stride equal to the number of ranks involved in the
        partitioning.  If a partition function is supplied via the 'func'
        argument, then the data will be partitioned across the 'worker' ranks,
        giving each 'worker' rank a different part of the data according to
        the algorithm used by partition function supplied.

        If the 'involved' argument is True, then a part of the data (as
        determined by the given partition function, if supplied) will be
        returned on the 'manager' rank.  Otherwise, ('involved' argument is
        False) the data will be partitioned only across the 'worker' ranks.

        This call must be made by all ranks.

        Keyword Arguments:
            data: The data to be partitioned across
                the ranks in the communicator.
            func: A PartitionFunction object/function that returns
                a part of the data given the index and assumed
                size of the partition.
            involved (bool): True, if a part of the data should be given
                to the 'manager' rank in addition to the 'worker'
                ranks. False, otherwise.
            tag (int): A user-defined integer tag to uniquely
                specify this communication message

        Returns:
            A (possibly partitioned) subset (i.e., part) of the data.
            Depending on the PartitionFunction used (or if it is used at all),
            this method may return a different part on each rank.
        """
        if self.is_manager():
            op = func if func else lambda *x: x[0][x[1] :: x[2]]
            j = 1 if not involved else 0
            for i in range(1, self.get_size()):

                # Get the part of the data to send to rank i
                part = op(data, i - j, self.get_size() - j)

                # Create the handshake message
                msg = {}
                msg['rank'] = self.get_rank()
                msg['buffer'] = self._is_bufferable(part)
                msg['shape'] = getattr(part, 'shape', None)
                msg['dtype'] = getattr(part, 'dtype', None)

                # Send the handshake message to the worker rank
                msg_tag = self._tag_offset(self.PART_TAG, self.MSG_TAG, tag)
                self._comm.send(msg, dest=i, tag=msg_tag)

                # Receive the acknowledgement from the worker
                ack_tag = self._tag_offset(self.PART_TAG, self.ACK_TAG, tag)
                ack = self._comm.recv(source=i, tag=ack_tag)

                # Check the acknowledgement, if bad skip this rank
                if not ack:
                    continue

                # If OK, send the data to the worker
                if msg['buffer']:
                    npy_tag = self._tag_offset(self.PART_TAG, self.NPY_TAG, tag)
                    self._comm.Send(self._numpy.array(part), dest=i, tag=npy_tag)
                else:
                    pyt_tag = self._tag_offset(self.PART_TAG, self.PYT_TAG, tag)
                    self._comm.send(part, dest=i, tag=pyt_tag)

            if involved:
                return op(data, 0, self.get_size())
            else:
                return None
        else:

            # Get the data message from the manager
            msg_tag = self._tag_offset(self.PART_TAG, self.MSG_TAG, tag)
            msg = self._comm.recv(source=0, tag=msg_tag)

            # Check the message content
            ack = type(msg) is dict and all(
                [key in msg for key in ['rank', 'buffer', 'shape', 'dtype']]
            )

            # If the message is good, acknowledge
            ack_tag = self._tag_offset(self.PART_TAG, self.ACK_TAG, tag)
            self._comm.send(ack, dest=0, tag=ack_tag)

            # if acknowledgement is bad, skip
            if not ack:
                return None

            # Receive the data
            if msg['buffer']:
                npy_tag = self._tag_offset(self.PART_TAG, self.NPY_TAG, tag)
                recvd = self._numpy.empty(msg['shape'], dtype=msg['dtype'])
                self._comm.Recv(recvd, source=0, tag=npy_tag)
            else:
                pyt_tag = self._tag_offset(self.PART_TAG, self.PYT_TAG, tag)
                recvd = self._comm.recv(source=0, tag=pyt_tag)

            return recvd

    def ration(self, data=None, tag=0):
        """
        Send a single piece of data from the 'manager' rank to a 'worker' rank.

        If this method is called on a 'worker' rank, the worker will send a
        "request" for data to the 'manager' rank.  When the 'manager' receives
        this request, the 'manager' rank sends a single piece of data back to
        the requesting 'worker' rank.

        For each call to this function on a given 'worker' rank, there must
        be a matching call to this function made on the 'manager' rank.

        NOTE: This method cannot be used for communication between the
        'manager' rank and itself.  Attempting this will cause the code to
        hang.

        Keyword Arguments:
            data: The data to be asynchronously sent to the 'worker' rank
            tag (int): A user-defined integer tag to uniquely specify this
                communication message

        Returns:
            On the 'worker' rank, the data sent by the manager.  On the
            'manager' rank, None.

        Raises:
            RuntimeError: If executed during a serial or 1-rank parallel run
        """
        if self.get_size() > 1:
            if self.is_manager():

                # Listen for a requesting worker rank
                req_tag = self._tag_offset(self.RATN_TAG, self.REQ_TAG, tag)
                rank = self._comm.recv(source=self._mpi.ANY_SOURCE, tag=req_tag)

                # Create the handshake message
                msg = {}
                msg['buffer'] = self._is_bufferable(data)
                msg['shape'] = data.shape if hasattr(data, 'shape') else None
                msg['dtype'] = data.dtype if hasattr(data, 'dtype') else None

                # Send the handshake message to the requesting worker
                msg_tag = self._tag_offset(self.RATN_TAG, self.MSG_TAG, tag)
                self._comm.send(msg, dest=rank, tag=msg_tag)

                # Receive the acknowledgement from the requesting worker
                ack_tag = self._tag_offset(self.RATN_TAG, self.ACK_TAG, tag)
                ack = self._comm.recv(source=rank, tag=ack_tag)

                # Check the acknowledgement, if not OK skip
                if not ack:
                    return

                # If OK, send the data to the requesting worker
                if msg['buffer']:
                    npy_tag = self._tag_offset(self.RATN_TAG, self.NPY_TAG, tag)
                    self._comm.Send(data, dest=rank, tag=npy_tag)
                else:
                    pyt_tag = self._tag_offset(self.RATN_TAG, self.PYT_TAG, tag)
                    self._comm.send(data, dest=rank, tag=pyt_tag)
            else:

                # Send a request for data to the manager
                req_tag = self._tag_offset(self.RATN_TAG, self.REQ_TAG, tag)
                self._comm.send(self.get_rank(), dest=0, tag=req_tag)

                # Receive the handshake message from the manager
                msg_tag = self._tag_offset(self.RATN_TAG, self.MSG_TAG, tag)
                msg = self._comm.recv(source=0, tag=msg_tag)

                # Check the message content
                ack = type(msg) is dict and all(
                    [key in msg for key in ['buffer', 'shape', 'dtype']]
                )

                # Send acknowledgement back to the manager
                ack_tag = self._tag_offset(self.RATN_TAG, self.ACK_TAG, tag)
                self._comm.send(ack, dest=0, tag=ack_tag)

                # If acknowledgement is bad, don't receive
                if not ack:
                    return None

                # Receive the data from the manager
                if msg['buffer']:
                    npy_tag = self._tag_offset(self.RATN_TAG, self.NPY_TAG, tag)
                    recvd = self._numpy.empty(msg['shape'], dtype=msg['dtype'])
                    self._comm.Recv(recvd, source=0, tag=npy_tag)
                else:
                    pyt_tag = self._tag_offset(self.RATN_TAG, self.PYT_TAG, tag)
                    recvd = self._comm.recv(source=0, tag=pyt_tag)
                return recvd
        else:
            err_msg = 'Rationing cannot be used in 1-rank parallel operation'
            raise RuntimeError(err_msg)

    def collect(self, data=None, tag=0):
        """
        Send data from a 'worker' rank to the 'manager' rank.

        If the calling MPI process is the 'manager' rank, then it will
        receive and return the data sent from the 'worker'.  If the calling
        MPI process is a 'worker' rank, then it will send the data to the
        'manager' rank.

        For each call to this function on a given 'worker' rank, there must
        be a matching call to this function made on the 'manager' rank.

        NOTE: This method cannot be used for communication between the
        'manager' rank and itself.  Attempting this will cause the code to
        hang.

        Keyword Arguments:
            data: The data to be collected asynchronously
                on the 'manager' rank.
            tag (int): A user-defined integer tag to uniquely
                specify this communication message

        Returns:
            tuple: On the 'manager' rank, a tuple containing the source rank
                ID and the the data collected.  None on all other ranks.

        Raises:
            RuntimeError: If executed during a serial or 1-rank parallel run
        """
        if self.get_size() > 1:
            if self.is_manager():

                # Receive the message from the worker
                msg_tag = self._tag_offset(self.CLCT_TAG, self.MSG_TAG, tag)
                msg = self._comm.recv(source=self._mpi.ANY_SOURCE, tag=msg_tag)

                # Check the message content
                ack = type(msg) is dict and all(
                    [key in msg for key in ['rank', 'buffer', 'shape', 'dtype']]
                )

                # Send acknowledgement back to the worker
                ack_tag = self._tag_offset(self.CLCT_TAG, self.ACK_TAG, tag)
                self._comm.send(ack, dest=msg['rank'], tag=ack_tag)

                # If acknowledgement is bad, don't receive
                if not ack:
                    return None

                # Receive the data
                if msg['buffer']:
                    npy_tag = self._tag_offset(self.CLCT_TAG, self.NPY_TAG, tag)
                    recvd = self._numpy.empty(msg['shape'], dtype=msg['dtype'])
                    self._comm.Recv(recvd, source=msg['rank'], tag=npy_tag)
                else:
                    pyt_tag = self._tag_offset(self.CLCT_TAG, self.PYT_TAG, tag)
                    recvd = self._comm.recv(source=msg['rank'], tag=pyt_tag)
                return msg['rank'], recvd

            else:

                # Create the handshake message
                msg = {}
                msg['rank'] = self.get_rank()
                msg['buffer'] = self._is_bufferable(data)
                msg['shape'] = data.shape if hasattr(data, 'shape') else None
                msg['dtype'] = data.dtype if hasattr(data, 'dtype') else None

                # Send the handshake message to the manager
                msg_tag = self._tag_offset(self.CLCT_TAG, self.MSG_TAG, tag)
                self._comm.send(msg, dest=0, tag=msg_tag)

                # Receive the acknowledgement from the manager
                ack_tag = self._tag_offset(self.CLCT_TAG, self.ACK_TAG, tag)
                ack = self._comm.recv(source=0, tag=ack_tag)

                # Check the acknowledgement, if not OK skip
                if not ack:
                    return

                # If OK, send the data to the manager
                if msg['buffer']:
                    npy_tag = self._tag_offset(self.CLCT_TAG, self.NPY_TAG, tag)
                    self._comm.Send(data, dest=0, tag=npy_tag)
                else:
                    pyt_tag = self._tag_offset(self.CLCT_TAG, self.PYT_TAG, tag)
                    self._comm.send(data, dest=0, tag=pyt_tag)
        else:
            err_msg = 'Collection cannot be used in a 1-rank communicator'
            raise RuntimeError(err_msg)

"""
data partitioning functions.

"""

from abc import ABCMeta, abstractmethod
from operator import itemgetter

class PartitionFunction(object):

    """
    The abstract base-class for all Partitioning Function objects.

    A PartitionFunction object is one with a __call__ method that takes
    three arguments.  The first argument is the data to be partitioned, the
    second argument is the index of the partition (or part) requested, and
    third argument is the number of partitions to assume when dividing
    the data.
    """

    __metaclass__ = ABCMeta

    @staticmethod
    def _check_types(data, index, size):
        """
        Check the types of the index and size arguments.

        Parameters:
            data: The data to be partitioned
            index (int): The index of the partition to return
            size (int): The number of partitions to make

        Raises:
            TypeError: The size or index arguments are not int
            IndexError: The size argument is less than 1, or the index
                argument is less than 0 or greater than or equal to size
        """

        # Check the type of the index
        if type(index) is not int:
            raise TypeError('Partition index must be an integer')

        # Check the value of index
        if index > size - 1 or index < 0:
            raise IndexError('Partition index out of bounds')

        # Check the type of the size
        if type(size) is not int:
            raise TypeError('Partition size must be an integer')

        # Check the value of size
        if size < 1:
            raise IndexError('Partition size less than 1 is invalid')

    @staticmethod
    def _is_indexable(data):
        """
        Check if the data object is indexable.

        Parameters:
            data: The data to be partitioned

        Returns:
            bool: True, if data is an indexable object. False, otherwise.
        """
        if hasattr(data, '__len__') and hasattr(data, '__getitem__'):
            return True
        else:
            return False

    @staticmethod
    def _are_pairs(data):
        """
        Check if the data object is an indexable list of pairs.

        Parameters:
            data: The data to be partitioned

        Returns:
            bool: True, if data is an indexable list of pairs.
                False, otherwise.
        """
        if PartitionFunction._is_indexable(data):
            return all(map(lambda i: PartitionFunction._is_indexable(i) and len(i) == 2, data))
        else:
            return False

    @abstractmethod
    def __call__(self):
        """
        Implements the partition algorithm.
        """
        return

class Duplicate(PartitionFunction):

    """
    Return a copy of the original input data in each partition.
    """

    def __call__(self, data, index=0, size=1):
        """
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Parameters:
            data: The data to be partitioned

        Keyword Arguments:
            index (int): A partition index into a part of the data
            size (int): The largest number of partitions allowed

        Returns:
            The indexed part of the data, assuming the data is divided into
            size parts.
        """
        self._check_types(data, index, size)

        return data

class EqualLength(PartitionFunction):

    """
    Partition an indexable object by striding through the data.

    The initial object is "chopped" along its length into roughly equal length
    sublists.  If the partition size is greater than the length of the input
    data, then it will return an empty list for 'empty' partitions.  If the
    data is not indexable, then it will return the data for index=0 only, and
    an empty list otherwise.
    """

    def __call__(self, data, index=0, size=1):
        """
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Parameters:
            data: The data to be partitioned

        Keyword Arguments:
            index (int): A partition index into a part of the data
            size (int): The largest number of partitions allowed

        Returns:
            The indexed part of the data, assuming the data is divided into
            size parts.
        """
        self._check_types(data, index, size)

        if self._is_indexable(data):
            (lenpart, remdata) = divmod(len(data), size)
            psizes = [lenpart] * size
            for i in range(remdata):
                psizes[i] += 1
            ibeg = 0
            for i in range(index):
                ibeg += psizes[i]
            iend = ibeg + psizes[index]
            return data[ibeg:iend]
        else:
            if index == 0:
                return [data]
            else:
                return []

class EqualStride(PartitionFunction):

    """
    Partition an object by chopping the data into roughly equal lengths.

    This returns a sublist of an indexable object by "striding" through the
    data in steps equal to the partition size.  If the partition size is
    greater than the length of the input data, then it will return an empty
    list for "empty" partitions.  If the data is not indexable, then it will
    return the data for index=0 only, and an empty list otherwise.
    """

    def __call__(self, data, index=0, size=1):
        """
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Parameters:
            data: The data to be partitioned

        Keyword Arguments:
            index (int): A partition index into a part of the data
            size (int): The largest number of partitions allowed

        Returns:
            The indexed part of the data, assuming the data is divided into
            size parts.
        """
        self._check_types(data, index, size)

        if self._is_indexable(data):
            if index < len(data):
                return data[index::size]
            else:
                return []
        else:
            if index == 0:
                return [data]
            else:
                return []


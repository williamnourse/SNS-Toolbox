Saving and Loading Networks
"""""""""""""""""""""""""""

For deploying an SNS-Network, it is often advantageous to compile the network as a model, and then save it for later
use. This is particularly important when compiling networks using the :code:`'sparse'` compiler option, as these
networks take long periods of time to compile. File structure is currently based around python's
`pickle format <https://docs.python.org/3/library/pickle.html>`_.

Saving a Network
=================

When saving networks, we call the :code:`save` method of :code:`sns_toolbox` and pass in the model and a desired file
name. The convention within SNS-Toolbox is to save compiled networks as :code:`.sns` files.
::

    import sns_toolbox
    from sns_toolbox.networks import Network

    net = Network()
    model = net.compile(backend='numpy', dt=0.1)

    # Save the network
    sns_toolbox.save(model, 'save_example.sns')

Loading a Network
==================

When loading a network, we can call the :code:`load` method of :code:`sns_toolbox`.
::
    model_loaded = sns_toolbox.load('save_example.sns')

Once the model is loaded, it behaves exactly the same as the original network, compiled to a specific backend.
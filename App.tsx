import React, {useEffect, useState, useContext, useReducer} from "react"
import {Switch, Route, Redirect, useHistory, useLocation} from "react-router-dom"
import Context, {EnableDragContext, MobileContext, SocketContext, ImagesContext, UpdateImagesContext, TextualInversionsContext,
HypernetworksContext, LorasContext, ReverseSortContext, NSFWImagesContext, ImageInputImagesContext} from "./Context"
import axios from "axios"
import {io} from "socket.io-client"
import functions from "./structures/Functions"
import MainPage from "./pages/MainPage"
import "./index.less"

require.context("./assets/icons", true)

const App: React.FunctionComponent = (props) => {
    const [ignored, forceUpdate] = useReducer(x => x + 1, 0)
    const [loaded, setLoaded] = useState(false)
    const [enableDrag, setEnableDrag] = useState(false)
    const [mobile, setMobile] = useState(false)
    const [socket, setSocket] = useState(null) as any
    const [images, setImages] = useState([])
    const [nsfwImages, setNSFWImages] = useState([])
    const [imageInputImages, setImageInputImages] = useState([])
    const [updateImages, setUpdateImages] = useState(true)
    const [textualInversions, setTextualInversions] = useState([])
    const [hypernetworks, setHypernetworks] = useState([])
    const [loras, setLoras] = useState([])
    const [reverseSort, setReverseSort] = useState(false)

    useEffect(() => {
        functions.preventDragging()
        setTimeout(() => {
            setLoaded(true)
        }, 100)
    }, [])

    const history = useHistory()
    const location = useLocation()

    useEffect(() => {
        setTimeout(() => {
            if (mobile) {
                if (enableDrag) functions.dragScroll(false)
                return
            }
            functions.dragScroll(enableDrag)
        }, 100)
    }, [enableDrag, mobile, history])

    useEffect(() => {
        const mobileQuery = (query: any) => {
            if (query.matches) {
                setMobile(true)
            } else {
                setMobile(false)
            }
        }
        const media = window.matchMedia("(max-width: 700px)")
        media.addEventListener("change", mobileQuery)
        mobileQuery(media)
        document.documentElement.style.visibility = "visible"
    }, [])

    useEffect(() => {
        const socket = io()
        socket.on("connect", () => {
            setSocket(socket)
            let interrogatorName = localStorage.getItem("interrogatorName")
            if (!interrogatorName) interrogatorName = "wdtagger"
            const modelName = localStorage.getItem("modelName")
            const vaeName = localStorage.getItem("vaeName")
            const clipSkip = localStorage.getItem("clipSkip")
            const processing = localStorage.getItem("processing")
            socket.emit("load interrogate model", interrogatorName)
            socket.emit("load diffusion model", modelName, vaeName, clipSkip, processing)
            socket.emit("load control models")
        })
        return () => {
            socket.disconnect()
        }
    }, [])

    const processImageUpdate = async () => {
        let images = await axios.get("/all-outputs").then((r) => r.data)
        setImages(images)
        let nsfwImages = await axios.get("/all-nsfw-outputs").then((r) => r.data)
        setNSFWImages(nsfwImages)
        let imageInputImages = await axios.get("/all-image-outputs").then((r) => r.data)
        setImageInputImages(imageInputImages)
    }

    useEffect(() => {
        if (updateImages) {
            processImageUpdate()
            setUpdateImages(false)
        }
    }, [updateImages])

    const processNetworkUpdate = async () => {
        const textualInversions = await axios.get("/textual-inversions").then((r) => r.data)
        setTextualInversions(textualInversions)
        const hypernetworks = await axios.get("/hypernetworks").then((r) => r.data)
        setHypernetworks(hypernetworks)
        const loras = await axios.get("/lora-models").then((r) => r.data)
        setLoras(loras)
    }

    useEffect(() => {
        processNetworkUpdate()
    }, [])

    return (
        <div className={`app ${!loaded ? "stop-transitions" : ""}`}>
            <ImageInputImagesContext.Provider value={{imageInputImages, setImageInputImages}}>
            <NSFWImagesContext.Provider value={{nsfwImages, setNSFWImages}}>
            <ReverseSortContext.Provider value={{reverseSort, setReverseSort}}>
            <LorasContext.Provider value={{loras, setLoras}}>
            <HypernetworksContext.Provider value={{hypernetworks, setHypernetworks}}>
            <TextualInversionsContext.Provider value={{textualInversions, setTextualInversions}}>
            <UpdateImagesContext.Provider value={{updateImages, setUpdateImages}}>
            <ImagesContext.Provider value={{images, setImages}}>
            <SocketContext.Provider value={{socket, setSocket}}>
            <MobileContext.Provider value={{mobile, setMobile}}>
            <EnableDragContext.Provider value={{enableDrag, setEnableDrag}}>
                <Context>
                    <Switch>
                        <Route exact path={["/", "/reverse-diffusion"]}><MainPage/></Route>
                        <Route path="*"><MainPage/></Route>
                    </Switch>
                </Context>
            </EnableDragContext.Provider>
            </MobileContext.Provider>
            </SocketContext.Provider>
            </ImagesContext.Provider>
            </UpdateImagesContext.Provider>
            </TextualInversionsContext.Provider>
            </HypernetworksContext.Provider>
            </LorasContext.Provider>
            </ReverseSortContext.Provider>
            </NSFWImagesContext.Provider>
            </ImageInputImagesContext.Provider>
        </div>
    )
}

export default App